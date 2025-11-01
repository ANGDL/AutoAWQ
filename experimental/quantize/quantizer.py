from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
import functools

import torch
import torch.nn as nn
from tqdm import tqdm
import transformers

from experimental.utils.logger import awq_logger as logger
from experimental.utils.quantization_progress import QuantizationProgressManager

from awq.quantize.quantizer import AwqQuantizer as BaseQuantizer
from awq.utils.utils import clear_memory, get_best_device
from awq.quantize.scale import apply_scale, apply_clip
from awq.utils.module import (
    append_str_prefix,
    get_op_name,
    get_named_linears,
    set_op_by_name,
    exclude_layers_to_not_quantize,
)


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
            progress_dir=None,
            enable_progress=True,
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
        progress_path = (
            Path(progress_dir).expanduser()
            if progress_dir
            else Path.cwd() / "awq_quant_progress"
        )

        self.progress_dir = progress_path if enable_progress else None
        self.layer_names = [get_op_name(self.model, module) for module in self.modules]
        if enable_progress:
            self.progress_manager = QuantizationProgressManager(self.progress_dir, logger=logger)
            self.progress_config = self.progress_manager.ensure_config(
                {
                    "w_bit": self.w_bit,
                    "group_size": self.group_size,
                    "zero_point": self.zero_point,
                    "version": self.version,
                    "apply_clip": self.apply_clip,
                    "act_bit": self.act_bit,
                    "module_count": len(self.modules),
                    "layer_order": self.layer_names,
                }
            )
            self._completed_layers = set(self.progress_config.get("completed_layers", []))
        else:
            self.progress_manager = None
            self.progress_config = {}
            self._completed_layers = set()

    def quantize(self):
        try:
            for i, module in enumerate(tqdm(self.modules, desc="AWQ")):
                layer_name = self.layer_names[i]
                cached_state = None
                if self.progress_manager is not None:
                    cached_state = self.progress_manager.load_layer_state(layer_name, i)
                    if cached_state is not None and not isinstance(cached_state, dict):
                        logger.warning(
                            "Ignoring malformed cached state for layer %s", layer_name
                        )
                        cached_state = None
                    setattr(self, "cached_state", cached_state)

                # Move module and inputs to correct device
                common_device = next(module.parameters()).device
                if common_device is None or str(common_device) == "cpu":
                    if torch.cuda.is_available():
                        best_device = "cuda:" + str(i % torch.cuda.device_count())
                    else:
                        best_device = get_best_device()

                    self.modules[i] = module.to(best_device)
                    module = self.modules[i]
                    common_device = next(module.parameters()).device

                if self.module_kwargs.get("position_ids") is not None:
                    self.module_kwargs["position_ids"] = self.module_kwargs[
                        "position_ids"
                    ].to(common_device)

                if self.module_kwargs.get("attention_mask") is not None:
                    self.module_kwargs["attention_mask"] = self.module_kwargs[
                        "attention_mask"
                    ].to(common_device)

                self.inps = self.inps.to(common_device)

                # We need to move the rotary embedding every time we move to a new module.
                # Transformers 4.45.0 moved rotary embedding to model definition as of this PR:
                # https://github.com/huggingface/transformers/pull/32617
                self.awq_model.move_embed(self.model, common_device)

                # Transformers >= 4.48.0 requires positional embeddings should be computed before forward pass
                if (
                    transformers.__version__ >= "4.48.0"
                    and self.module_kwargs.get("position_embeddings") is None and hasattr(self.model.model, "rotary_emb")
                ):
                    self.module_kwargs["position_embeddings"] = self.model.model.rotary_emb(
                        self.inps, self.module_kwargs["position_ids"]
                    )

                if (
                    transformers.__version__ >= "4.48.0"
                    and self.module_kwargs.get("attention_mask") is None
                ):
                    self.module_kwargs["attention_mask"] = None

                for k, v in self.module_kwargs.items():
                    # position embeddings found in tuple
                    if isinstance(v, tuple):
                        self.module_kwargs[k] = tuple(
                            item.to(common_device)
                            if isinstance(item, (torch.Tensor, nn.Module))
                            else item
                            for item in v
                        )

                # [STEP 1]: Get layer, extract linear modules, extract input features
                named_linears = get_named_linears(module)

                # Filter out the linear layers we don't want to exclude
                not_converted_layers = []
                named_linears = exclude_layers_to_not_quantize(
                    named_linears, self.modules_to_not_convert, not_converted_layers
                )
                self.not_converted_layers.extend(
                    append_str_prefix(not_converted_layers, layer_name + ".")
                )

                input_feat = self._get_input_feat(module, named_linears)
                clear_memory()

                # [STEP 2]: Compute/apply scale list or reuse cached artifacts
                restored = False
                scales_list_local: List[Tuple[str, Tuple[str, ...], torch.Tensor]] = []
                clip_list_local: List[Tuple[str, torch.Tensor]] = []

                if cached_state and cached_state.get("scales_list"):
                    restored = True
                    scales_list_local = list(cached_state.get("scales_list", []))
                    clip_list_local = list(cached_state.get("clip_list", []))
                    apply_scale(module, scales_list_local, input_feat_dict=input_feat)
                    if self.apply_clip and clip_list_local:
                        apply_clip(module, clip_list_local)
                    logger.debug(
                        f"Restored quantization artifacts for layer {layer_name}"
                    )
                else:
                    module_config: List[Dict] = self.awq_model.get_layers_for_scaling(
                        module, input_feat, self.module_kwargs
                    )
                    scales_list_local = [
                        self._search_best_scale(module, **layer)
                        for layer in module_config
                    ]
                    apply_scale(module, scales_list_local, input_feat_dict=input_feat)

                    if self.apply_clip:
                        clip_list_local = self._search_best_clip(
                            module, named_linears, input_feat
                        )
                        apply_clip(module, clip_list_local)

                if not self.apply_clip:
                    clip_list_local = []

                # [STEP 4]: Quantize weights
                if not self.export_compatible:
                    self._apply_quant(module, named_linears, input_feat_dict=input_feat)

                metadata = dict(cached_state.get("metadata", {})) if cached_state else {}
                metadata.update(
                    {
                        "source": "restored" if restored else "computed",
                        "apply_clip": self.apply_clip,
                        "export_compatible": self.export_compatible,
                    }
                )

                if self.progress_manager is not None:
                    self.progress_manager.save_layer_async(
                        layer_name=layer_name,
                        layer_index=i,
                        scales_list=scales_list_local,
                        clip_list=clip_list_local,
                        extra_metadata=metadata,
                    )
                    self.progress_manager.mark_layer_done(layer_name)
                    self._completed_layers.add(layer_name)
                clear_memory()
        finally:
            if self.progress_manager is not None:
                self.progress_manager.close()

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