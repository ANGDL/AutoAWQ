import json
import os
from typing import Dict, List


class ConfigUpdater:
    """
    A class to update the quantization configuration in a model's config.json file.
    Supports 'w8a8_compressed' and 'w4a8' quantization types.
    'w8a8_compressed' use for vllm.
    'w4a8' use for sglang.
    """
    def __init__(self, awq_config: Dict, config_type: str = "w8a8_compressed"):
        self.awq_config = awq_config
        self.config_type = config_type
        
    def _create_w8a8_compressed_config(self, not_converted_layers: List[str]):
        ignore = list(set(not_converted_layers))
        w_bits = self.awq_config.get("w_bit", 8)
        assert w_bits in [8], "Only 8-bit weight quantization is supported for w8a8_compressed."
        w_strategy = "group" if self.awq_config.get("q_group_size", 128) != -1 else "channel"
        assert w_strategy in ["group", "channel"], "q_group_size should be -1 or a positive integer."

        config_dict = {
            "global_compression_ratio": 1.0,
            "format": "int-quantized",
            "ignore": ignore,
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
            "kv_cache_scheme": None,
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": w_bits,
                        "strategy": w_strategy,
                        "symmetric": True,
                        "type": "int",
                        "observer": "minmax",
                        "observer_kwargs": {},
                        "dynamic": False,
                        "group_size": self.awq_config.get("q_group_size", None),
                    },
                    "input_activations": {
                        "dynamic": True,
                        "num_bits": 8,
                        "observer": None,
                        "strategy": "token",
                        "symmetric": True,
                        "type": "int"
                    },
                }
            },
        }

        return config_dict

    def _create_w4a8_config(self, not_converted_layers: List[str]):
        w_bits = self.awq_config.get("w_bit", 8)
        assert w_bits in [4], "Only 4-bit weight quantization is supported for W4A8."
        config_dict = {
            "quant_method": "w4a8",
            "self_attn_w8a8": False,
            "modules_to_not_convert": list(set(not_converted_layers))
        }
        return config_dict


    def update_config(self, save_dir: str, not_converted_layers: List[str] = []):
        config_file = f"{save_dir}/config.json"
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found at {config_file}")

        with open(config_file, "r") as f:
            config = json.load(f)

        w_bits = self.awq_config.get("w_bit", None)

        if self.config_type == "w8a8_compressed" and w_bits == 8:
            config["quantization_config"] = self._create_w8a8_compressed_config(not_converted_layers)
        elif self.config_type == "w4a8" and w_bits == 4:
            config["quantization_config"] = self._create_w4a8_config(not_converted_layers)
        else:
            raise ValueError(f"Unsupported config_type {self.config_type} or w_bit {w_bits}")

        with open(config_file, "w") as f:
            json.dump(config, f, indent=4, ensure_ascii=False, sort_keys=True)
    
    def __repr__(self):
        return f"ConfigUpdater(config_type={self.config_type}, awq_config={self.awq_config})"
