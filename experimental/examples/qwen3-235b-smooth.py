import logging
from awq import AutoAWQForCausalLM
from experimental.quantize.smoother import SmoothQuantizer
from transformers import AutoTokenizer
import torch
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.DEBUG)

"""
Usage:
First,
    run "python experimental/examples/qwen3-235b-smooth.py" will get INT8 weights for Qwen3-235B-A22B-Instruct-2507 model.
Then,
    In the config.json file, change the quantization_config to use the llm-compressor's quantization_config.
"""

model_path = '/data/models/Qwen3-235B-A22B-Instruct-2507'
quant_path = '/klxlake/personal/zhuang/models/Qwen3-235B-A22B-Instruct-2507-awq-w8a8-4'

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048


quant_config = {"zero_point": False, "q_group_size": -1, "w_bit": 8, "version": "GEMM", "modules_to_not_convert":["mlp.gate"]}

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, low_cpu_mem_usage=True, use_cache=False)

print(model.model)

# Quantize
model.quantize(
    tokenizer, 
    quant_config=quant_config, 
#    calib_data=calib_data, 
#    act_bit=8,
    max_calib_seq_len=MAX_SEQUENCE_LENGTH,
    max_calib_samples=NUM_CALIBRATION_SAMPLES,
    quantizer_cls=SmoothQuantizer,
    )

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
