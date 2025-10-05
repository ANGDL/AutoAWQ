from typing import Dict, List, Optional

import torch
import torch.nn as nn
from awq.utils.utils import clear_memory, get_best_device
from awq.utils.module import (
    get_op_name,
    set_op_by_name,
)
from experimental.quantize.quantizer import MoeFixedQuantizer as BaseQuantizer
from experimental.modules.linear.gemm import WQ8Linear_GEMM


class SmoothQuantizer(BaseQuantizer):
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
        only_smooth=False
    ):
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
            only_smooth
        )

    @torch.no_grad()
    def _search_best_scale(
        self,
        module,
        prev_op,
        layers: List[nn.Linear],
        inp: torch.Tensor,
        module2inspect=None,
        kwargs={},
    ):
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        if "use_cache" in kwargs:
            kwargs.pop("use_cache")

        # Put x on the right device
        inp = inp.to(next(module2inspect.parameters()).device)

        # [STEP 1]: Compute per-channel max value of normalised weights
        # All layer weights are concatted together
        weight = torch.cat([_m.weight for _m in layers], dim=0)
        # The weights are reshaped to be organised by quantization group
        if self.group_size > 0:
            weight = weight.view(-1, self.group_size)

        w_scale = weight.abs_().amax(dim=0).clamp(min=1e-4)
        clear_memory(weight)

        # [STEP 2]: Compute per-channel mean of the input activation with chunking
        # move inp to cpu to avoid memory leak
        inp_flat = inp.cpu().abs().view(-1, inp.shape[-1])
            
        num_elements = inp_flat.size(0)
        num_channels = inp_flat.size(1)
        element_size_bytes = inp_flat.element_size() * 2 # multiplied by 2 for FP32

        # Calculate chunk size dynamically based on max_chunk_memory
        chunk_size = int(self.max_chunk_memory // (element_size_bytes * num_channels))
        chunk_size = min(chunk_size, num_elements)

        # Use float32 for max value
        x_scale = inp_flat[0].to(torch.float32).to(inp.device)  # Initialize with correct shape
        
        for i in range(0, num_elements, chunk_size):
            end = min(i + chunk_size, num_elements)
            chunk_max = inp_flat[i:end].to(torch.float32).amax(dim=0)
            x_scale = torch.max(x_scale, chunk_max.to(inp.device))

        x_scale = x_scale.to(weight.dtype).clamp(min=1e-4)

        # [STEP 3]: Compute output of module
        with torch.no_grad():
            module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)
            fp16_output = self._module_forward(inp, module2inspect, module_kwargs)
            fp16_output = fp16_output.clip(torch.finfo(fp16_output.dtype).min, torch.finfo(fp16_output.dtype).max)
        
        # [STEP 4]: Compute loss
        if self.act_bit is not None:
            # Register a forward_pre_hook to quantize Linear layer input activations
            def quantize_input_hook(module, input):
                x = input[0]
                x_shape = x.shape
                x_q, _, _ = self.pseudo_quantize_tensor(x.reshape(-1, x_shape[-1]), self.group_size, w_bit=self.act_bit)
                return (x_q.reshape(x_shape),) + input[1:]

            hook_handles = []
            try:
                for fc in layers:
                    handle = fc.register_forward_pre_hook(quantize_input_hook)
                    hook_handles.append(handle)

                best_scales = self._compute_best_scale(
                    inp, w_scale, x_scale, module2inspect, layers, fp16_output, module_kwargs
                )
            finally:
                for handle in hook_handles:
                    handle.remove()
        else:
            best_scales = self._compute_best_scale(
                inp, w_scale, x_scale, module2inspect, layers, fp16_output, module_kwargs
            )
            
        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            best_scales,
        )
    
    def pseudo_quantize_tensor(self, w: torch.Tensor, group_size, zero_point=None, w_bit=None):
        if zero_point is None:
            zero_point = self.zero_point
        if w_bit is None:
            w_bit = self.w_bit
        org_w_shape = w.shape
        if group_size > 0:
            assert org_w_shape[-1] % group_size == 0, f"org_w_shape ({org_w_shape[-1]}) must be a multiple of group_size ({group_size})!"
            w = w.reshape(-1, group_size)
        assert w.dim() == 2
        assert torch.isnan(w).sum() == 0

        # zero point quantization
        if zero_point:
            max_val = w.amax(dim=1, keepdim=True)
            min_val = w.amin(dim=1, keepdim=True)
            max_int = 2**w_bit - 1
            min_int = 0
            scales = (max_val - min_val).clamp(min=1e-5) / max_int
            zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
            w = (
                torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
            ) * scales
            zeros = zeros.view(org_w_shape[0], -1)
        else:
            max_val = w.abs().amax(dim=1, keepdim=True)
            max_val = max_val.clamp(min=1e-5)
            max_int = 2 ** (w_bit - 1) - 1
            min_int = -(2 ** (w_bit - 1))
            scales = max_val / max_int
            zeros = None
            w = torch.clamp(torch.round(w / scales), min_int, max_int) * scales

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        scales = scales.view(org_w_shape[0], -1)
        w = w.reshape(org_w_shape)

        return w, scales, zeros
    
    def _apply_quant(self, module, named_linears: Dict[str, nn.Linear]):
        for name, linear_layer in named_linears.items():
            # NOTE: small regression in perplexity if linear layer uses .cpu().float()
            linear_layer = linear_layer.to(get_best_device()).half()

            linear_layer.weight.data, scales, zeros = self.pseudo_quantize_tensor(
                linear_layer.weight.data,
                self.group_size
            )

            if self.version == "gemm":
                scales = scales.contiguous()
                if zeros is not None:
                    zeros = zeros.contiguous()
                q_linear_module = WQ8Linear_GEMM

            else:
                raise ValueError(f"Unknown version {self.version}")

            q_linear = q_linear_module.from_linear(
                linear=linear_layer,
                w_bit=self.w_bit,
                group_size=self.group_size if self.group_size > 0 else linear_layer.weight.data.shape[-1],
                init_only=False,
                scales=scales,
                zeros=zeros,
            )

            linear_layer.cpu()
            q_linear.to(next(module.parameters()).device)
            set_op_by_name(module, name, q_linear)
            clear_memory()
