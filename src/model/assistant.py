import torch
from transformers import LlamaForCausalLM, GenerationConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
from ..config import LoRAConfig
import warnings
import os

# Filter out specific warnings
warnings.filterwarnings("ignore", message=".*_register_pytree_node.*")
warnings.filterwarnings("ignore", message=".*MatMul8bitLt: inputs will be cast.*")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class AssistantModel(torch.nn.Module):
    def __init__(self, base_model_path: str, num_layers: int, load_in_8bit: bool, lora_config: LoRAConfig):
        super().__init__()  # Call parent constructor
        self.device = torch.device("cuda:0")
        self.model = self._create_base_model(base_model_path, num_layers, load_in_8bit)
        self.model = self._add_lora_layers(self.model, lora_config)
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

    def _create_base_model(self, model_path: str, num_layers: int, load_in_8bit: bool) -> LlamaForCausalLM:
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=load_in_8bit,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            use_cache=False,  # Explicitly disable cache for gradient checkpointing
        )
        
        # Keep only first n layers
        model.model.layers = model.model.layers[:num_layers]
        
        if load_in_8bit:
            model = prepare_model_for_kbit_training(model)\

        # Enable gradients for all parameters
        for param in model.parameters():
            if param.dtype in [torch.float16, torch.float32, torch.float64, 
                            torch.complex64, torch.complex128, torch.bfloat16]:
                param.requires_grad = True
            
        # Prepare model for training even when not using 8-bit
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

            
        return model
    
    def _add_lora_layers(self, model: LlamaForCausalLM, config: LoRAConfig) -> LlamaForCausalLM:
        lora_config = LoraConfig(
            r=config.r,
            lora_alpha=config.alpha,
            target_modules=config.target_modules,
            lora_dropout=config.dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        model = get_peft_model(model, lora_config)

        # Double check all trainable parameters have requires_grad=True
        for name, param in model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True

        return model
    
    def parameters(self):
        """Return model parameters for optimizer."""
        return self.model.parameters()
    
    def train(self, mode = True):
        """Set model to training mode."""
        self.model.train(mode)
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        
    def to(self, device):
        """Move model to device."""
        self.model = self.model.to(device)
        return self
        
    def __call__(self, *args, **kwargs):
        """Forward pass through model."""
        with torch.amp.autocast('cuda'):
            output = self.model(*args, **kwargs)
        # torch.cuda.synchronize()
        return output
    
    def save_pretrained(self, save_path: str):
        """Save the model and its configuration."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save the LoRA adapter weights and config
        self.model.save_pretrained(save_path)
        
        # Save the base model configuration
        if hasattr(self.model, 'config'):
            self.model.config.save_pretrained(save_path)
            
        # Save the model path for later loading
        with open(os.path.join(save_path, "model_path.txt"), "w") as f:
            f.write(self.model.base_model.name_or_path)
            
    def load_pretrained(self, load_path: str):
        """Load the model and its configuration."""
        # Load the original model path
        with open(os.path.join(load_path, "model_path.txt"), "r") as f:
            base_model_path = f.read().strip()
            
        # Create new base model
        base_model = self._create_base_model(
            base_model_path, 
            num_layers=len(self.model.base_model.model.layers),
            load_in_8bit=False
        )
        
        # Load the LoRA adapter
        self.model = PeftModel.from_pretrained(
            base_model,
            load_path
        )

    def state_dict(self):
        return self.model.state_dict()
        
    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Generate text using the model."""
        self.eval()  # Ensure model is in eval mode
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        generation_config = self.generation_config
        for k, v in kwargs.items():
            setattr(generation_config, k, v)
        
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config
        )