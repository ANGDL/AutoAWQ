import tqdm
from typing import List, Tuple, Union, Optional, Dict
from .base import BaseAWQForCausalLM
import torch
from transformers.cache_utils import Cache


class MiniMaxM2AWQForCausalLM(BaseAWQForCausalLM):
    layer_type = "MiniMaxM2DecoderLayer"
    max_seq_len_key = "max_position_embeddings"

    @staticmethod
    def get_model_layers(model):
        return model.model.layers

    @staticmethod
    def get_act_for_scaling(module):
        return dict(is_scalable=False)

    @staticmethod
    def move_embed(model, device: str):
        model.model.embed_tokens = model.model.embed_tokens.to(device)

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
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            layers.append(
                dict(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )

        # linear in
        layers.append(
            dict(
                prev_op=module.post_attention_layernorm,
                layers=[
                    w
                    for expert in module.block_sparse_moe.experts
                    for w in [expert.w1, expert.w3]
                ],
                inp=input_feat["block_sparse_moe"],
                module2inspect=module.block_sparse_moe,
            )
        )

        # linear out
        for i, expert in enumerate(module.block_sparse_moe.experts):
            layers.append(
                dict(
                    prev_op=expert.w3,
                    layers=[expert.w2],
                    inp=input_feat[f"block_sparse_moe.experts.{i}.w2"],
                )
            )

        return layers

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -past_key_values.get_seq_length() - 1 :]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
        }


