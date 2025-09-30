import torch
import warnings
import torch.nn as nn


class WQ8Linear_GEMM(nn.Module):
    def __init__(
        self, w_bit, group_size, in_features, out_features, bias, dev, training=False
    ):
        super().__init__()

        if w_bit not in [8]:
            raise NotImplementedError("Only 8-bit are supported for WQ8Linear_GEMM.")
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.training = training

        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0

        self.register_buffer(
            "weight",
            torch.zeros(
                (out_features, in_features),
                dtype=torch.int8,
                device=dev,
            ),
        )
        self.register_buffer(
            "weight_scale",
            torch.zeros(
                (out_features, in_features // self.group_size),
                dtype=torch.float16,
                device=dev,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (out_features),
                    dtype=torch.float16,
                    device=dev,
                ),
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None
    ):
        # need scales info for real quantization
        assert scales is not None, "scales should not be None for WQ8Linear_GEMM"
        assert zeros is None, "zeros should be None for WQ8Linear_GEMM"

        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd
            return awq_linear

        awq_linear.weight_scale[:] = scales.half()
        if linear.bias is not None:
            awq_linear.bias[:] = linear.bias.half()

        awq_linear.weight[:] = (linear.weight.data / awq_linear.weight_scale).round().to(torch.int8)
        return awq_linear

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)

        input_dtype = x.dtype
        if input_dtype != torch.float16:
            x = x.half()

        if self.training:
            out = (x @ (self.weight.float() * self.weight_scale)).half()
        else:
            with torch.no_grad():
                out = (x @ (self.weight.float() * self.weight_scale)).half()

        if self.bias is not None:
            out = out + self.bias

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)

        return out.reshape(out_shape)

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.w_bit,
                self.group_size,
            )
        )
