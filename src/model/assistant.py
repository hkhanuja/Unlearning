import torch
from transformers import LlamaForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
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
        self.model = self._create_base_model(base_model_path, num_layers, load_in_8bit)
        self.model = self._add_lora_layers(self.model, lora_config)
    
    def _create_base_model(self, model_path: str, num_layers: int, load_in_8bit: bool) -> LlamaForCausalLM:
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=load_in_8bit,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            use_cache=False,  # Explicitly disable cache for gradient checkpointing
        )
        
        # Keep only first n layers
        model.model.layers = model.model.layers[:num_layers]
        
        if load_in_8bit:
            model = prepare_model_for_kbit_training(model)\

        # Enable gradients for all parameters
        for param in model.parameters():
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
        self.model.save_pretrained(save_path)
        
    def load_pretrained(self, load_path: str):
        """Load the model and its configuration."""
        self.model = self.model.from_pretrained(load_path)

    def state_dict(self):
        return self.model.state_dict()
        
    def load_state_dict(self, state_dict):
        return self.model.load_state_dict(state_dict)