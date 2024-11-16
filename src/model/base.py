import torch
from transformers import LlamaForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit_fp32_cpu_offload=True)

class BaseModel:
    def __init__(self, model_path: str, load_in_8bit: bool):
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=load_in_8bit,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            quantization_config = quantization_config
        )
        self.model.eval()