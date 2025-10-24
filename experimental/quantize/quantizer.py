from collections import defaultdict
from typing import Dict
import functools

import torch

from awq.quantize.quantizer import AwqQuantizer as BaseQuantizer
from awq.utils.utils import clear_memory
from experimental.utils.logger import awq_logger as logger


class ExpandedQuantizer(BaseQuantizer):
    def __init__(
            self, 
            awq_model, 
            model, 
            tokenizer, 
            w_bit, 
            group_size, 
            zero_point, 
            version, 
            calib_data, 
            split, 
            text_column, 
            duo_scaling, 
            modules_to_not_convert=None, 
            export_compatible=False, 
            apply_clip=True, 
            n_parallel_calib_samples=None, 
            max_calib_samples=128, 
            max_calib_seq_len=512, 
            max_chunk_memory=1024 * 1024 * 1024, 
            act_bit=None, 
            **kwargs
            ) -> None:
        super().__init__(
            awq_model, 
            model, 
            tokenizer, 
            w_bit, 
            group_size, 
            zero_point, 
            version, 
            calib_data, 
            split, 
            text_column, 
            duo_scaling, 
            modules_to_not_convert, 
            export_compatible, 
            apply_clip, 
            n_parallel_calib_samples, 
            max_calib_samples, 
            max_calib_seq_len, 
            max_chunk_memory, 
            act_bit, 
            )

    def _get_input_feat(self, layer, named_linears):
        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []

        # FIXME: Workaround for Mixtral to use block_sparse_moe input features
        if self.awq_model.model_type == "mixtral":
            named_linears = {
                **named_linears,
                "block_sparse_moe": layer.block_sparse_moe,
            }

        if self.awq_model.model_type == "deepseek_v2" or self.awq_model.model_type == "deepseek_v3":
            named_linears = {
                **named_linears,
                "mlp": layer.mlp,
            }
        
        if self.awq_model.model_type == "qwen3_moe":
            named_linears = {
                **named_linears,
                "mlp": layer.mlp,
            }

        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(
                        cache_input_hook, 
                        name=name, 
                        feat_dict=input_feat, 
                    )
                )
            )
        self.inps = self.inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input

        # Sanitize the kwargs in case we use transformers version that contains
        # kwargs that are not handled by the module.
        # Useful for trust_remote_code models.
        module_kwargs = self._sanitize_kwargs(self.module_kwargs, layer)

        self.inps = self._module_forward(self.inps, layer, module_kwargs)

        # check all expert layers are visited
        # some experts are not activated, enable use input feature from other experts
        unactivated_experts = list(set(named_linears.keys()) - set(input_feat.keys()))
        unactivated_experts += list(k for k, v in input_feat.items() if len(v) == 0 or v[0].shape[0] == 0)
        if len(unactivated_experts) > 0:
            logger.warning(f"Some experts are not activated: {unactivated_experts}, will use input features from other experts with the same layer name suffix.")
            activated_experts = set(named_linears.keys()) - set(unactivated_experts)

            for expert in unactivated_experts:
                for name in activated_experts:
                    if name.split('.')[-1] == expert.split('.')[-1]:
                        input_feat[expert] = [torch.ones_like(input_feat[name][0])]
                        logger.warning(f"Using all-one input feature for unactivated expert {expert}, as it shares the same shape as {name}.")
                        break
                    
        for h in handles:
            h.remove()

        # now solve for scaling and clipping
        def cat_and_assert(k, v):
            x = torch.cat(v, dim=0)
            assert x.shape[0] != 0, (
                f"{k} has a zero dimension. This can happen if no data was passed through (e.g. an expert in MoE not being activated). "
                "Try increasing max_calib_samples (warning: this can significantly increase quantization time and memory usage.)"
            )
            return x

        input_feat = {k: cat_and_assert(k, v) for k, v in input_feat.items()}

        return input_feat
    
    @torch.no_grad()
    def _module_forward(
        self, x: torch.Tensor, module: torch.nn.Module, module_kwargs: Dict
    ) -> torch.Tensor:
        try:
            return super()._module_forward(x, module, module_kwargs)
        except torch.OutOfMemoryError as e:
            if self.n_parallel_calib_samples is None or self.n_parallel_calib_samples > 1:
                self.n_parallel_calib_samples = 1
            clear_memory(force=True)
            return super()._module_forward(x, module, module_kwargs)
        
        except Exception as e:
            logger.error(f"Unexpected error during module forward: {e}")
            raise e