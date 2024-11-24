import warnings

# Filter warnings
warnings.filterwarnings("ignore", message=".*_register_pytree_node*")
warnings.filterwarnings("ignore", message=".*MatMul8bitLt: inputs will be cast.*")
warnings.filterwarnings("ignore", message=".*gradient checkpointing.*")
warnings.filterwarnings("ignore", message=".*torch.load*")

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
from src.model.base import BaseModel
from src.tokenizer import UnlearningTokenizer

def calculate_metrics(original_responses: List[str], 
                     unlearned_responses: List[str], 
                     ground_truth: List[str],
                     logger: logging.Logger) -> Dict:
    """Calculate evaluation metrics with logging."""
    from nltk.translate.bleu_score import sentence_bleu
    from rouge import Rouge
    
    logger.info("Calculating evaluation metrics...")
    rouge = Rouge()
    
    # First, let's check for empty responses
    empty_responses = []
    for i, (orig, unl, gt) in enumerate(zip(original_responses, unlearned_responses, ground_truth)):
        if not orig.strip() or not unl.strip() or not gt.strip():
            empty_responses.append(i)
            logger.warning(f"Empty response found at index {i}:")
            logger.warning(f"Original: '{orig}'")
            logger.warning(f"Unlearned: '{unl}'")
            logger.warning(f"Ground truth: '{gt}'")
    
    if empty_responses:
        logger.warning(f"Found {len(empty_responses)} empty responses. Filtering them out.")
        # Filter out empty responses
        filtered_orig = [r for i, r in enumerate(original_responses) if i not in empty_responses]
        filtered_unl = [r for i, r in enumerate(unlearned_responses) if i not in empty_responses]
        filtered_gt = [r for i, r in enumerate(ground_truth) if i not in empty_responses]
    else:
        filtered_orig = original_responses
        filtered_unl = unlearned_responses
        filtered_gt = ground_truth
    
    if not filtered_orig:
        logger.error("No valid responses remaining after filtering!")
        return {
            "privacy_protection": {
                "bleu": 0.0,
                "rouge": 0.0
            },
            "utility_preservation": {
                "bleu": 0.0,
                "rouge": 0.0
            }
        }

    try:
        # Add minimal content for empty responses to avoid ROUGE errors
        processed_unl = [unl if unl.strip() else "empty" for unl in filtered_unl]
        processed_orig = [orig if orig.strip() else "empty" for orig in filtered_orig]
        processed_gt = [gt if gt.strip() else "empty" for gt in filtered_gt]

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
    
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        logger.error("Sample responses:")
        for i in range(min(5, len(filtered_unl))):
            logger.error(f"Index {i}:")
            logger.error(f"Original: '{filtered_orig[i]}'")
            logger.error(f"Unlearned: '{filtered_unl[i]}'")
            logger.error(f"Ground truth: '{filtered_gt[i]}'")
        raise

def main():
    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = r"/home/hice1/pli319/scratch/CSE8803/logs"
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
        downloader = ModelDownloader(r"/home/hice1/pli319/scratch/CSE8803/models")
        model_path, _ = downloader.download_alpaca()
        
        model_config = ModelConfig(
            model_path=model_path,
            num_assistant_layers=8,
            load_in_8bit=False  # Disable 8-bit quantization
        )
        
        # Initialize models with proper device handling
        logger.info("Initializing models...")
        try:
            # Try loading original model
            logger.info("Loading original model...")
            base_model = BaseModel(
                model_path=model_config.model_path,
                load_in_8bit=False  
            )
            
            # Initialize tokenizer
            tokenizer = UnlearningTokenizer(model_config.model_path)
            
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
                    # Format prompt properly
                    formatted_prompt = f"### Instruction:\n{item['prompt']}\n\n### Response:\n"
                    logger.debug(f"Processing prompt: {formatted_prompt}")

                    # Generate with base model
                    inputs = tokenizer(
                        formatted_prompt,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True
                    ).to("cuda:0")
                    
                    # Generate with memory clearing between batches
                    with torch.amp.autocast('cuda'):  # Use automatic mixed precision
                        base_outputs = base_model.generate(
                            **inputs,
                            max_length=512,
                            temperature=0.7,
                            output_scores=True,
                            return_dict_in_generate=True,
                        )
                        original_response = tokenizer.decode(base_outputs.sequences[0], skip_special_tokens=True)
                        

                        unlearned_response = unlearned_model.generate(item["prompt"])
                        
                        # Add validation
                        if not original_response.strip():
                            logger.warning(f"Empty original response for prompt: {item['prompt']}")
                        if not unlearned_response.strip():
                            logger.warning(f"Empty unlearned response for prompt: {item['prompt']}")

                    original_responses.append(original_response)
                    unlearned_responses.append(unlearned_response)
                    ground_truth.append(item["response"])
                    
                    # Clear cache after each generation
                    torch.cuda.empty_cache()

                if i > 20 * batch_size:
                    break

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