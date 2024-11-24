import torch
from transformers import LlamaForCausalLM, BitsAndBytesConfig, GenerationConfig

# quantization_config = BitsAndBytesConfig(load_in_8bit_fp32_cpu_offload=True)

class BaseModel:
    def __init__(self, model_path: str, load_in_8bit: bool):
        self.model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=load_in_8bit,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            # quantization_config = quantization_config
        )
        self.model.eval()
        self.device = torch.device("cuda:0")
        self.model = self.model.to(self.device)

        # Create generation config
        self.generation_config = GenerationConfig(
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.model.config.pad_token_id,
            bos_token_id=self.model.config.bos_token_id,
            eos_token_id=self.model.config.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        **kwargs
    ):
        """Generate text using the base model."""
        self.model.eval()
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)


        generation_config = self.generation_config
        for k, v in kwargs.items():
            setattr(generation_config, k, v)

        try:
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config
            )
            return outputs
            
        except Exception as e:
            print(f"Error in base model generation: {str(e)}")
            raise