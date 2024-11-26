import os
import json
import torch
from typing import Optional, Dict, Any

# Import model components
from transformers import GenerationConfig
from src.config import ModelConfig, LoRAConfig
from src.model.base import BaseModel
from src.model.assistant import AssistantModel
from src.tokenizer import UnlearningTokenizer
from src.generator import ResponseGenerator
from utils.model_utils import ModelDownloader

class UnlearnedModelWrapper:
    def __init__(self, base_model_path: str, assistant_model_path: str = None):
        """
        Initialize the unlearned model wrapper.
        
        Args:
            base_model_path: Path to the base model
            assistant_model_path: Path to the saved assistant model (with LoRA weights)
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Initialize base components
        self.tokenizer = UnlearningTokenizer(base_model_path)
        self.base_model = BaseModel(base_model_path, load_in_8bit=False)
        
        # Initialize config
        model_config = ModelConfig(
            model_path=base_model_path,
            num_assistant_layers=8,
            load_in_8bit=False
        )
        
        # Initialize assistant model if path provided
        if assistant_model_path:
            self.assistant_model = AssistantModel(
                base_model_path,
                model_config.num_assistant_layers,
                load_in_8bit=False,
                lora_config=LoRAConfig()
            )
            # Load the saved assistant model
            self.assistant_model.load_pretrained(assistant_model_path)
        else:
            self.assistant_model = None
            
        # Initialize generator
        self.generator = ResponseGenerator(
            self.base_model,
            self.assistant_model,
            self.tokenizer
        )
        
    def to(self, device):
        """Move model to specified device."""
        self.device = torch.device(device)
        self.base_model = self.base_model.to(device)
        if self.assistant_model:
            self.assistant_model = self.assistant_model.to(device)
        return self
        
    def eval(self):
        """Set model to evaluation mode."""
        self.base_model.eval()
        if self.assistant_model:
            self.assistant_model.eval()
        return self
        
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        alpha: float = 0.75,
        **kwargs
    ) -> str:
        """
        Generate text using the unlearned model.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            alpha: Weight for logit subtraction
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        return self.generator.generate(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            alpha=alpha,
            **kwargs
        )
    
    @staticmethod
    def download_base_model(model_name: str = "chavinlo/alpaca-native", cache_dir: str = "models", force_download: bool = False) -> str:
        """
        Download the base model and return its path.
        
        Args:
            model_name: HuggingFace model identifier
            cache_dir: Directory to store downloaded models
            force_download: Whether to force re-download even if files exist
            
        Returns:
            Path to the downloaded model
        """
        try:
            downloader = ModelDownloader(cache_dir=cache_dir)
            model_path, _ = downloader.download_alpaca(
                model_name=model_name,
                force_download=force_download
            )
            return model_path
        except Exception as e:
            raise RuntimeError(f"Error downloading base model: {str(e)}")

    @classmethod
    def from_pretrained(cls, base_model_path: str = None, assistant_model_path: str = None, 
                       download_model: bool = False, model_name: str = "chavinlo/alpaca-native", 
                       cache_dir: str = "models", **kwargs):
        """
        Load a pretrained unlearned model.
        
        Args:
            base_model_path: Path to base model (if None and download_model=True, will download)
            assistant_model_path: Path to assistant model
            download_model: Whether to download the base model if path not provided
            model_name: HuggingFace model identifier for downloading
            cache_dir: Directory to store downloaded models
            **kwargs: Additional loading parameters
            
        Returns:
            UnlearnedModelWrapper instance
        """
        if base_model_path is None and download_model:
            base_model_path = cls.download_base_model(
                model_name=model_name,
                cache_dir=cache_dir
            )
        elif base_model_path is None:
            raise ValueError("Must provide either base_model_path or set download_model=True")
            
        return cls(base_model_path, assistant_model_path)
    
    def save_pretrained(self, save_path: str):
        """
        Save the unlearned model.
        
        Args:
            save_path: Directory to save the model
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Save assistant model if it exists
        if self.assistant_model:
            assistant_path = os.path.join(save_path, "assistant_model")
            self.assistant_model.save_pretrained(assistant_path)
            
        # Save model configuration
        config = {
            "base_model_path": self.base_model.model.name_or_path,
            "assistant_model_path": os.path.join(save_path, "assistant_model") if self.assistant_model else None
        }
        
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)


def main():
    # Download the base model
    base_model_path = UnlearnedModelWrapper.download_base_model(
        model_name="chavinlo/alpaca-native",
        cache_dir="models",
        force_download=False
    )

    # Load the model
    model = UnlearnedModelWrapper.from_pretrained(
        base_model_path=base_model_path,
        assistant_model_path="CSE8803/models/best_model.pt"
    )

    # Move to device if needed
    model = model.to("cuda:0")

    # Generate text
    response = model.generate(
        "What is the capital of France?",
        max_length=512,
        temperature=0.7
    )

    print(response)

if __name__ == "__main__":
    main()