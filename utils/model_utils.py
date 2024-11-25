import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import torch
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class ModelDownloader:
    def __init__(self, cache_dir: str = "models"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def download_alpaca(
        self,
        model_name: str = "chavinlo/alpaca-native",
        force_download: bool = False
    ) -> Tuple[str, str]:
        """
        Download Alpaca model and tokenizer, returning their paths.
        
        Args:
            model_name: Hugging Face model identifier
            force_download: Whether to force re-download even if files exist
            
        Returns:
            Tuple of (model_path, tokenizer_path)
        """
        return self._download_model(model_name, force_download)
    
    def download_llama_8b_instruct(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        force_download: bool = False
    ) -> Tuple[str, str]:
        """
        Download LLaMA-8B-Instruct model and tokenizer, returning their paths.
        
        Args:
            model_name: Hugging Face model identifier
            force_download: Whether to force re-download even if files exist
            
        Returns:
            Tuple of (model_path, tokenizer_path)
        """
        return self._download_model(model_name, force_download)
    
    def _download_model(self, model_name: str, force_download: bool) -> Tuple[str, str]:
        """
        Generic method for downloading a Hugging Face model and tokenizer.
        
        Args:
            model_name: Hugging Face model identifier
            force_download: Whether to force re-download even if files exist
            
        Returns:
            Tuple of (model_path, tokenizer_path)
        """
        logger.info(f"Downloading model: {model_name}")
        
        try:
            # Create model-specific cache directory
            model_cache_dir = os.path.join(self.cache_dir, model_name.split('/')[-1])
            os.makedirs(model_cache_dir, exist_ok=True)
            
            if not force_download and os.path.exists(model_cache_dir) and len(os.listdir(model_cache_dir)) > 0:
                logger.info(f"Model already exists in {model_cache_dir}")
                return model_cache_dir, model_cache_dir
            
            # Download model and tokenizer
            logger.info("Downloading model files...")
            snapshot_download(
                repo_id=model_name,
                cache_dir=self.cache_dir,
                local_dir=model_cache_dir,
                ignore_patterns=["*.msgpack", "*.h5", "*.txt"]
            )
            
            # Verify downloads by loading
            logger.info("Verifying downloaded files...")
            tokenizer = AutoTokenizer.from_pretrained(model_cache_dir)
            model = AutoModelForCausalLM.from_pretrained(
                model_cache_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                use_cache=False,
                weights_only=True
            )
            
            logger.info(f"Successfully downloaded model to {model_cache_dir}")
            return model_cache_dir, model_cache_dir
            
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            raise
    
    def verify_model_files(self, model_path: str) -> bool:
        """Verify that all necessary model files exist."""
        required_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
        
        for file in required_files:
            if not os.path.exists(os.path.join(model_path, file)):
                return False
        return True
