import torch
import torch.nn as nn
from typing import Tuple, List, Union
import logging
from awq.utils.utils import get_best_device
from awq.modules.act import ScaledActivation
from awq.utils.module import get_op_by_name, set_op_by_name
from awq.quantize import scale
from transformers.models.bloom.modeling_bloom import BloomGelu
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
from transformers.models.gemma2.modeling_gemma2 import Gemma2RMSNorm
from transformers.models.cohere.modeling_cohere import CohereLayerNorm
from transformers.activations import NewGELUActivation, PytorchGELUTanh, GELUActivation


logger = logging.getLogger(__name__)

# Reuse the canonical allowed_* definitions from awq.quantize.scale
# so experimental logic stays in sync and avoids duplication.
allowed_norms = getattr(scale, "allowed_norms", [])
allowed_act_fns = getattr(scale, "allowed_act_fns", [])

def check_and_clip_tensor(tensor: torch.Tensor, tensor_name: str):
    """
    Check if the tensor values exceed its data type range and clip them if necessary.
    """
    dtype = tensor.dtype
    if dtype.is_floating_point:
        min_val, max_val = torch.finfo(dtype).min, torch.finfo(dtype).max
    else:
        min_val, max_val = torch.iinfo(dtype).min, torch.iinfo(dtype).max

    if torch.any(tensor < min_val) or torch.any(tensor > max_val):
        logger.warning(f"Warning: {tensor_name} exceeded data type range. Clipping values.")
        tensor.clamp_(min_val, max_val)


def apply_scale(module, scales_list, input_feat_dict=None):
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        best_device = get_best_device()
        prev_op.to(best_device)
        for layer in layers:
            layer.to(best_device)
        scales.to(best_device)

        if (
            isinstance(prev_op, nn.Linear)
            and type(layers) == list
            and isinstance(layers[0], nn.Linear)
        ):
            scale_fc_fcs(prev_op, layers, scales)

        elif isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales)

        elif (
            any(isinstance(prev_op, t) for t in allowed_norms)
            or "rmsnorm" in str(prev_op.__class__).lower()
        ):
            scale_ln_fcs(prev_op, layers, scales)

        elif any(isinstance(prev_op, t) for t in allowed_act_fns):
            new_module = ScaledActivation(prev_op, scales)
            set_op_by_name(module, prev_op_name, new_module)
            scale_gelu_fc(prev_op, layers[0], scales)

        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        # apply the scaling to input feat if given; prepare it for clipping
        if input_feat_dict is not None:
            for layer_name in layer_names:
                # Skip the modules that are not quantized
                if layer_name in input_feat_dict:
                    inp = input_feat_dict[layer_name]
                    inp.div_(scales.view(1, -1).to(inp.device))
                    check_and_clip_tensor(inp, f"input_feat_dict[{layer_name}]")

        prev_op.cpu()
        for layer in layers:
            layer.cpu()
        scales.cpu()


@torch.no_grad()
def scale_ln_fcs(ln: nn.Linear, fcs: List[nn.Linear], scales: torch.Tensor):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    # GemmaRMSNorm is different from Llama's in that it multiplies
    # (1 + weight) to the output, instead of just weight.
    if isinstance(ln, GemmaRMSNorm) or isinstance(ln, Gemma2RMSNorm):
        ln.weight += 1
        ln.weight.div_(scales)
        check_and_clip_tensor(ln.weight, "ln.weight")
        ln.weight -= 1
    else:
        ln.weight.div_(scales)
        check_and_clip_tensor(ln.weight, "ln.weight")

    if hasattr(ln, "bias") and ln.bias is not None:
        ln.bias.div_(scales)
        check_and_clip_tensor(ln.bias, "ln.bias")

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
        check_and_clip_tensor(fc.weight, "fc.weight")

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1: nn.Linear, fc2: nn.Linear, scales: torch.Tensor):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)

    scales = scales.to(fc1.weight.device)

    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    check_and_clip_tensor(fc1.weight, "fc1.weight")
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))
        check_and_clip_tensor(fc1.bias, "fc1.bias")

    fc2.weight.mul_(scales.view(1, -1))
    check_and_clip_tensor(fc2.weight, "fc2.weight")

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fcs(fc1: nn.Linear, fcs: List[nn.Linear], scales: torch.Tensor):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(fc1.weight.device)

    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
    check_and_clip_tensor(fc1.weight, "fc1.weight")
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))
        check_and_clip_tensor(fc1.bias, "fc1.bias")

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
        check_and_clip_tensor(fc.weight, "fc.weight")

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_gelu_fc(gelu: Union[nn.Module, type], fc: nn.Linear, scales: torch.Tensor):
    assert any(isinstance(gelu, t) for t in allowed_act_fns)
    assert isinstance(fc, nn.Linear)

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
    check_and_clip_tensor(fc.weight, "fc.weight")

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0

scale.apply_scale = apply_scale