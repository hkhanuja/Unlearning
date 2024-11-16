# evaluate.py
import os
from datetime import datetime
from src.config import ModelConfig, LoRAConfig, TrainingConfig
from src.unlearning import AlpacaUnlearning
from utils.data_utils import load_data, load_model
from utils.model_utils import ModelDownloader
from utils.logger import setup_logger
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import json
import logging
import torch

def calculate_metrics(original_responses: List[str], 
                     unlearned_responses: List[str], 
                     ground_truth: List[str],
                     logger: logging.Logger) -> Dict:
    """Calculate evaluation metrics with logging."""
    from nltk.translate.bleu_score import sentence_bleu
    from rouge import Rouge
    
    logger.info("Calculating evaluation metrics...")
    rouge = Rouge()
    
    # Privacy Protection
    privacy_bleu = np.mean([
        sentence_bleu([orig.split()], unl.split())
        for orig, unl in zip(original_responses, unlearned_responses)
    ])
    
    privacy_rouge = np.mean([
        rouge.get_scores(unl, orig)[0]['rouge-l']['f']
        for orig, unl in zip(original_responses, unlearned_responses)
    ])
    
    # Utility Preservation
    utility_bleu = np.mean([
        sentence_bleu([gt.split()], unl.split())
        for gt, unl in zip(ground_truth, unlearned_responses)
    ])
    
    utility_rouge = np.mean([
        rouge.get_scores(unl, gt)[0]['rouge-l']['f']
        for gt, unl in zip(ground_truth, unlearned_responses)
    ])
    
    metrics = {
        "privacy_protection": {
            "bleu": float(privacy_bleu),
            "rouge": float(privacy_rouge)
        },
        "utility_preservation": {
            "bleu": float(utility_bleu),
            "rouge": float(utility_rouge)
        }
    }
    
    logger.info("Metrics calculation completed")
    return metrics

def main():
    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(
        "evaluation",
        log_file=f"{log_dir}/evaluation_{timestamp}.log"
    )
    
    try:
        logger.info("Loading model paths...")
        with open("outputs/latest_model.txt", "r") as f:
            unlearned_model_path = f.read().strip()
        
        # Download/load base model with memory-efficient settings
        logger.info("Setting up models...")
        downloader = ModelDownloader()
        model_path, _ = downloader.download_alpaca()
        
        model_config = ModelConfig(
            model_path=model_path,
            num_assistant_layers=8,
            load_in_8bit=True  # Enable 8-bit quantization
        )
        
        # Initialize models with proper device handling
        logger.info("Initializing models...")
        try:
            # Try loading original model
            logger.info("Loading original model...")
            original_model = AlpacaUnlearning(
                model_config=model_config,
                lora_config=LoRAConfig(),
                training_config=TrainingConfig()
            )
            
            # Try loading unlearned model
            logger.info("Loading unlearned model...")
            unlearned_model = AlpacaUnlearning(
                model_config=model_config,
                lora_config=LoRAConfig(),
                training_config=TrainingConfig()
            )
            load_model(unlearned_model, unlearned_model_path)
            
            # Process test data in smaller batches
            logger.info("Loading test data...")
            test_data = load_data("data/Privacy Violation_test.csv")
            batch_size = 4  # Smaller batch size for evaluation
            
            logger.info("Generating responses...")
            original_responses = []
            unlearned_responses = []
            ground_truth = []
            
            # Process in batches
            for i in tqdm(range(0, len(test_data), batch_size)):
                batch = test_data[i:min(i + batch_size, len(test_data))]
                
                for item in batch:
                    # Generate with memory clearing between batches
                    with torch.cuda.amp.autocast():  # Use automatic mixed precision
                        original_response = original_model.generate(item["prompt"])
                        unlearned_response = unlearned_model.generate(item["prompt"])
                        
                    original_responses.append(original_response)
                    unlearned_responses.append(unlearned_response)
                    ground_truth.append(item["response"])
                    
                    # Clear cache after each generation
                    torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"Error during model operations: {str(e)}")
            raise
            
        
        # Calculate metrics
        metrics = calculate_metrics(
            original_responses,
            unlearned_responses,
            ground_truth,
            logger
        )
        
        # Save results
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f"evaluation_results_{timestamp}.json")
        
        with open(results_path, 'w') as f:
            json.dump({
                "metrics": metrics,
                "model_path": unlearned_model_path,
                "timestamp": timestamp
            }, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        # Print results
        logger.info("\nEvaluation Results:")
        logger.info("Privacy Protection (lower is better):")
        logger.info(f"  BLEU: {metrics['privacy_protection']['bleu']:.4f}")
        logger.info(f"  ROUGE: {metrics['privacy_protection']['rouge']:.4f}")
        logger.info("\nUtility Preservation (higher is better):")
        logger.info(f"  BLEU: {metrics['utility_preservation']['bleu']:.4f}")
        logger.info(f"  ROUGE: {metrics['utility_preservation']['rouge']:.4f}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()