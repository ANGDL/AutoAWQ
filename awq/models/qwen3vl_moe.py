import tqdm
from typing import List, Tuple

from torch import nn
import torch

from .base import BaseAWQForCausalLM
from awq.utils.utils import skip_weights_initialize

from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import (
        Qwen3VLMoeTextSparseMoeBlock,
        Qwen3VLMoeTextMLP
    )


class AWQForQwen3VLMoeTextSparseMoeBlock(nn.Module):
    def __init__(
        self,
        original: "Qwen3VLMoeTextSparseMoeBlock",
        config: "Qwen3VLMoeConfig",
        calibrate_all_experts: bool,
    ):
        super().__init__()
        text_config: "Qwen3VLMoeTextConfig" = config.get_text_config()

        self.hidden_size = text_config.hidden_size
        self.num_experts = text_config.num_experts
        self.top_k = original.top_k
        # Note: gate was changed to be a Linear layer in transformers==4.57.0
        # https://github.com/JJJYmmm/transformers/commit/f5dea1c694af8c994c769170813a8702332119ee
        self.gate = original.gate
        self.calibrate_all_experts = calibrate_all_experts
        self.experts = SequentialQwen3VLMoeTextExperts(text_config, original.experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(
            router_logits, dim=1, dtype=torch.float
        )
        # get topk experts per token
        # routing_weight: (num_tokens, top_k)
        # routing_indices: (num_tokens, top_k)
        routing_weights, router_indices = torch.topk(
            routing_weights, self.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        next_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # convert router indices into OHE list
        # reshape to be (num_experts, top_k, batch_size * sequence_length)
        expert_mask = torch.nn.functional.one_hot(
            router_indices, num_classes=self.num_experts
        ).permute(2, 1, 0)

        for expert_idx, expert_layer in enumerate(self.experts):
            idx, token_idx = torch.where(expert_mask[expert_idx].squeeze(0))

            if self.calibrate_all_experts:
                expert_out = expert_layer(hidden_states)[token_idx]
            else:
                expert_out = expert_layer(hidden_states[token_idx])

            if len(token_idx) > 0:
                # if there are tokens meant for this expert, further scale the expert
                # output by the score
                weighted_output = expert_out * routing_weights[token_idx, idx, None]
                next_states.index_add_(
                    0, token_idx, weighted_output.to(hidden_states.dtype)
                )

        next_states = next_states.reshape(batch_size, sequence_length, hidden_dim)
        return next_states, router_logits


class SequentialQwen3VLMoeTextExperts(torch.nn.ModuleList):
    def __init__(self, config, original):
        self.num_experts = original.gate_up_proj.shape[0]
        with skip_weights_initialize():
            super().__init__(
                [Qwen3VLMoeTextMLP(config) for _ in range(self.num_experts)]
            )

        intermediate_size = original.down_proj.shape[1]

        for i in range(self.num_experts):
            gate_up = original.gate_up_proj[i]
            down = original.down_proj[i]

            gate_proj = gate_up[:, :intermediate_size]
            up_proj = gate_up[:, intermediate_size:]

            self[i].gate_proj.weight.data = gate_proj.t().clone().contiguous()
            self[i].up_proj.weight.data = up_proj.t().clone().contiguous()
            self[i].down_proj.weight.data = down.t().clone().contiguous()


class Qwen3VLMoeAWQForConditionalGeneration(BaseAWQForCausalLM):
    layer_type = "Qwen3VLMoeTextDecoderLayer"
    max_seq_len_key = "max_position_embeddings"
    modules_to_not_convert = ["visual"]

    @staticmethod
    def get_model_layers(model):
        return model.model.language_model.layers

    @staticmethod
    def get_act_for_scaling(module):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model, device: str):
        model.model.language_model.embed_tokens = model.model.language_model.embed_tokens.to(device)
        model.model.language_model.rotary_emb = model.model.language_model.rotary_emb.to(device)
        model.model.visual = model.model.visual.to(device)
        for i, layer in enumerate(model.model.language_model.layers):
            if isinstance(layer.mlp, Qwen3VLMoeTextSparseMoeBlock):
                layer.mlp = AWQForQwen3VLMoeTextSparseMoeBlock(layer.mlp, model.config, calibrate_all_experts=False)

    @staticmethod
    def get_layers_for_scaling(module, input_feat, module_kwargs):
        layers = []

        # attention input
        layers.append(
            dict(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn,
                kwargs=module_kwargs,
            )
        )

        # attention out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )

        if hasattr(module.mlp, "gate"):
            # linear in
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    layers=[
                        w
                        for expert in module.mlp.experts
                        for w in [expert.gate_proj, expert.up_proj]
                    ],
                    inp=input_feat["mlp"],
                    module2inspect=module.mlp,
                )
            )

            # linear out
            for i, expert in enumerate(module.mlp.experts):
                layers.append(
                    dict(
                        prev_op=expert.up_proj,
                        layers=[expert.down_proj],
                        inp=input_feat[f"mlp.experts.{i}.down_proj"],
                    )
                )

        else:
            # linear 1
            layers.append(
                dict(
                    prev_op=module.post_attention_layernorm,
                    layers=[module.mlp.gate_proj, module.mlp.up_proj],
                    inp=input_feat["mlp.gate_proj"],
                    module2inspect=module.mlp,
                )
            )

            # linear 2
            layers.append(
                dict(
                    prev_op=module.mlp.up_proj,
                    layers=[module.mlp.down_proj],
                    inp=input_feat["mlp.down_proj"],
                )
            )

        return layers