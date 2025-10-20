import os
import math
import time
from typing import Optional, Tuple, Dict, Callable, List
from tqdm import tqdm
from functools import partial

import torch
import transformers
from torch import nn

from experimental.quantize.smoother import SmoothQuantizer
from experimental.modules.linear.gemm import WQ8Linear_GEMM
from awq.utils.utils import clear_memory, get_best_device
from awq.utils.module import (
    append_str_prefix,
    get_op_name,
    set_op_by_name,
    get_named_linears,
    exclude_layers_to_not_quantize
)
from awq.quantize.scale import apply_scale, apply_clip
from experimental.utils.logger import awq_logger as logger

__all__ = ["GPTQQuantizer"]


GPTQ_PRECISION = torch.float32


def _quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


class _Quantizer(nn.Module):
    def __init__(self, shape=1):
        super(_Quantizer, self).__init__()
        self.register_buffer("maxq", torch.tensor(0))
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

    def configure(
        self,
        bits,
        perchannel=False,
        sym=True,
        mse=False,
        norm=2.4,
        grid=100,
        maxshrink=0.8,
        trits=False,
    ):
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            self.scale = (xmax - xmin) / self.maxq
            if self.sym:
                self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
            else:
                self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float("inf"), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = _quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return _quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)
    

class GPTQ:
    def __init__(
            self, 
            inp: torch.Tensor, 
            out: torch.Tensor,
            module: torch.nn.Module, 
            batch_size: int = 1, 
            loss_func: Optional[Callable] = None
            ):
        self.inp = inp
        self.out = out
        self.num_samples = 0
        self.layer = module
        self.H = self.make_empty_hessian(module, inp.device)
        for i in range(0, inp.shape[0], batch_size):
            end = min(i + batch_size, inp.shape[0])
            self.H, self.num_samples = self.accumulate_hessian(
                inp[i:end], module, self.H, self.num_samples
            )
        assert self.num_samples > 0, "No samples were used to compute the Hessian."
        self.columns = self.H.shape[0]
        self.dev = module.weight.device
        self.quantizer = _Quantizer()
        self.loss_func = loss_func if loss_func is not None else nn.MSELoss(reduction="mean")

    @staticmethod
    def make_empty_hessian(
        module: torch.nn.Module, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        weight = module.weight
        num_columns = weight.shape[1]
        device = device if device is not None else weight.device
        return torch.zeros((num_columns, num_columns), device=device, dtype=GPTQ_PRECISION)

    @staticmethod
    def accumulate_hessian(
        inp: torch.Tensor,
        module: torch.nn.Module,
        H: Optional[torch.Tensor],
        num_samples: int,
    ) -> Tuple[torch.Tensor, int]:
        inp = inp.to(device=H.device)
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)

        num_added = inp.shape[0]

        if isinstance(module, (torch.nn.Linear, transformers.Conv1D)):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        if isinstance(module, torch.nn.Conv2d):
            unfold = torch.nn.Unfold(
                module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

        H *= num_samples / (num_samples + num_added)
        num_samples += num_added

        inp = inp.to(dtype=GPTQ_PRECISION)
        inp = math.sqrt(2 / num_samples) * inp
        H += inp.matmul(inp.t())

        return H, num_samples

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        group_size=-1,
        actorder=False,
        static_groups=False,
    ):
        assert self.quantizer is not None and self.quantizer.enabled(), "Quantizer must be provided and enabled."
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        g_idx = []
        scale = []
        zero = []
        now_idx = 1

        if static_groups:
            import copy

            groups = []
            for i in range(0, self.columns, group_size):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i : (i + group_size)], weight=True)
                scale.append(quantizer.scale)
                zero.append(quantizer.zero)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if group_size != -1:
                    if not static_groups:
                        if (i1 + i) % group_size == 0:
                            self.quantizer.find_params(W[:, (i1 + i) : (i1 + i + group_size)], weight=True)

                        if ((i1 + i) // group_size) - now_idx == -1:
                            scale.append(self.quantizer.scale)
                            zero.append(self.quantizer.zero)
                            now_idx += 1
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // group_size]

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            # if os.environ.get("AWQ_DEBUG"):
            #     self.layer.weight.data[:, :i2] = Q[:, :i2].to(self.layer.weight.dtype)
            #     self.layer.weight.data[:, i2:] = W[:, i2:].to(self.layer.weight.dtype)
            #     logger.debug(f"[Block {i1},{i2}], GPTQ Loss: {torch.sum(Losses) :<.6f}, AWQ Loss: {self.loss_func(self.out, self.layer(self.inp)) :<.7f}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif torch.xpu.is_available():
            torch.xpu.synchronize()
        elif torch.mps.is_available():
            torch.mps.synchronize()

        logger.debug(f"duration: {time.time() - tick :<.4f} sec")
        logger.debug(f"AVG GPTQ Loss: {torch.sum(Losses).item() / self.num_samples :<.7f}")

        group_size = group_size if group_size != -1 else self.columns
        if static_groups and actorder:
            g_idx = [perm[i] // group_size for i in range(self.columns)]
        else:
            g_idx = [i // group_size for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        if actorder:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).type_as(self.layer.weight.data)

        pred_loss = self.loss_func(self.out, self.layer(self.inp))

        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)
        return scale, zero, g_idx, pred_loss

    def free(self):
        if os.environ.get("AWQ_DEBUG"):
            self.inp = None
            self.out = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()

class GPTQQuantizer(SmoothQuantizer):
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
        **kwargs,
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
        )
        self.batch_size = kwargs.get("batch_size", 1)
        self.block_size = kwargs.get("block_size", 128)
        self.perc_damp = kwargs.get("perc_damp", 0.01)
        self.act_order = kwargs.get("act_order", False)
        self.static_groups = kwargs.get("static_groups", False)

        assert not self.act_order, "act_order is not supported in this implementation."

    @torch.no_grad()
    def _quantize_impl(
        self,
        name: str,
        linear_layer: nn.Linear,
        input_feat: Optional[torch.Tensor],
    ):
        if input_feat is None:
            raise ValueError(f"GPTQ requires input features for layer {name}, but got None.")
        
        inp = input_feat.to(linear_layer.weight.dtype).to(linear_layer.weight.device)
        out = linear_layer(inp)
        weight_copy = linear_layer.weight.data.cpu()

        q_weight_awq, q_scales, q_zeros = self.pseudo_quantize_tensor(
            linear_layer.weight.data,
            self.group_size
        )
        linear_layer.weight.data = q_weight_awq.to(linear_layer.weight.dtype).to(linear_layer.weight.device)
        qout = linear_layer(inp)
        q_loss = self._compute_loss(out, qout, device=linear_layer.weight.device)    

        linear_layer.weight.data = weight_copy.to(linear_layer.weight.dtype).to(linear_layer.weight.device)
        gptq = GPTQ(
            inp,
            out,
            linear_layer,
            batch_size=self.batch_size,
            loss_func=partial(self._compute_loss, device=linear_layer.weight.device),
        )
        gptq.quantizer.configure(
            self.w_bit,
            perchannel=(self.group_size != -1),
            sym=not self.zero_point,
            mse=True,
            trits=(self.w_bit not in [4, 8]),
        )
        gptq_scales, gptq_zeros, _, gptq_loss = gptq.fasterquant(
            blocksize=self.block_size,
            percdamp=self.perc_damp,
            group_size=self.group_size,
            actorder=self.act_order,
            static_groups=self.static_groups,
        )
        gptq.free()

        if q_loss < gptq_loss:
            logger.debug(f"Layer {name}: AWQ loss {q_loss:<.7f} is lower than GPTQ loss {gptq_loss:<.7f}. Using AWQ quantization.")
            linear_layer.weight.data = q_weight_awq.to(linear_layer.weight.dtype).to(linear_layer.weight.device)
            scales = q_scales
            zeros = q_zeros
        else:
            logger.debug(f"Layer {name}: GPTQ loss {gptq_loss:<.7f} is lower than AWQ loss {q_loss:<.7f}. Using GPTQ quantization.")
            scales = gptq_scales
            zeros = gptq_zeros

        if not self.zero_point:
            zeros = None

        return scales, zeros