# evaluate.py
import os
from datetime import datetime
from typing import List, Dict
import torch
import numpy as np
from tqdm import tqdm
import json
import logging
from pathlib import Path

from src.config import ModelConfig, LoRAConfig, TrainingConfig
from src.unlearning import AlpacaUnlearning
from utils.data_utils import load_data, save_model
from utils.model_utils import ModelDownloader
from utils.logger import setup_logger
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

class UnlearningEvaluator:
    def __init__(self, base_model_path: str, assistant_model_path: str, 
                 alpha: float = 0.75, filter_threshold: float = 1e-2):
        self.logger = setup_logger("evaluator", 
                                 f"logs/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # Initialize models
        self.logger.info("Initializing models...")
        model_config = ModelConfig(
            model_path=base_model_path,
            num_assistant_layers=8,
            load_in_8bit=False  # Using this since it worked before
        )
        
        # Initialize base system
        self.unlearner = AlpacaUnlearning(
            model_config=model_config,
            lora_config=LoRAConfig(),
            training_config=TrainingConfig()
        )
        
        # Load trained assistant model
        self.logger.info(f"Loading assistant model from {assistant_model_path}")
        self.unlearner.assistant_model.load_pretrained(assistant_model_path)
        
        self.alpha = alpha
        self.filter_threshold = filter_threshold
        
    def generate_with_logit_subtraction(self, prompt: str, max_length: int = 512) -> str:
        """Generate text using logit subtraction between base and assistant models."""
        formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
        inputs = self.unlearner.tokenizer.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        ).to("cuda:0")
        
        with torch.no_grad():
            # Get base model logits
            base_outputs = self.unlearner.base_model.model.generate(
                **inputs,
                max_length=max_length,
                output_scores=True,
                return_dict_in_generate=True,
                temperature=0.7,
            )
            
            # Get assistant model logits
            assistant_outputs = self.unlearner.assistant_model.model.generate(
                **inputs,
                max_length=max_length,
                output_scores=True,
                return_dict_in_generate=True,
                temperature=0.7,
            )
            
            # Apply logit subtraction with filtering
            filtered_scores = []
            for base_score, assistant_score in zip(base_outputs.scores, assistant_outputs.scores):
                # Apply threshold filter to assistant logits
                mask = torch.abs(assistant_score) > self.filter_threshold
                filtered_assistant = assistant_score * mask
                
                # Subtract filtered assistant logits from base logits
                final_score = base_score - self.alpha * filtered_assistant
                filtered_scores.append(final_score)
            
            # Convert scores to tokens
            final_tokens = torch.cat([
                torch.argmax(score, dim=-1) for score in filtered_scores
            ])
            
            return self.unlearner.tokenizer.tokenizer.decode(final_tokens, skip_special_tokens=True)
    
    def evaluate_dataset(self, test_data: List[Dict], batch_size: int = 4) -> Dict:
        """Evaluate the model on test dataset."""
        self.logger.info("Starting evaluation...")
        
        metrics = {
            "original_responses": [],
            "unlearned_responses": [],
            "ground_truth": [],
            "privacy_metrics": {},
            "utility_metrics": {}
        }
        
        # Process in batches
        for i in tqdm(range(0, len(test_data), batch_size)):
            batch = test_data[i:min(i + batch_size, len(test_data))]
            
            for item in batch:
                # Generate responses
                with torch.cuda.amp.autocast():
                    # Original response (without logit subtraction)
                    original_response = self.unlearner.base_model.generate(
                        item["prompt"], 
                        max_length=512
                    )
                    
                    # Unlearned response (with logit subtraction)
                    unlearned_response = self.generate_with_logit_subtraction(
                        item["prompt"]
                    )
                    
                metrics["original_responses"].append(original_response)
                metrics["unlearned_responses"].append(unlearned_response)
                metrics["ground_truth"].append(item["response"])
                
                # Clear cache
                torch.cuda.empty_cache()
        
        # Calculate metrics
        rouge = Rouge()
        
        # Privacy protection metrics
        metrics["privacy_metrics"] = {
            "bleu": np.mean([
                sentence_bleu([orig.split()], unl.split())
                for orig, unl in zip(metrics["original_responses"], 
                                   metrics["unlearned_responses"])
            ]),
            "rouge": np.mean([
                rouge.get_scores(unl, orig)[0]['rouge-l']['f']
                for orig, unl in zip(metrics["original_responses"], 
                                   metrics["unlearned_responses"])
            ])
        }
        
        # Utility preservation metrics
        metrics["utility_metrics"] = {
            "bleu": np.mean([
                sentence_bleu([gt.split()], unl.split())
                for gt, unl in zip(metrics["ground_truth"], 
                                 metrics["unlearned_responses"])
            ]),
            "rouge": np.mean([
                rouge.get_scores(unl, gt)[0]['rouge-l']['f']
                for gt, unl in zip(metrics["ground_truth"], 
                                 metrics["unlearned_responses"])
            ])
        }
        
        return metrics
    
    def save_results(self, metrics: Dict, output_dir: str = "results"):
        """Save evaluation results."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
        with open(results_path, 'w') as f:
            json.dump({
                "metrics": metrics,
                "config": {
                    "alpha": self.alpha,
                    "filter_threshold": self.filter_threshold
                },
                "timestamp": timestamp
            }, f, indent=2)
        
        self.logger.info(f"Results saved to {results_path}")
        
        # Log summary metrics
        self.logger.info("\nEvaluation Results:")
        self.logger.info("Privacy Protection (lower is better):")
        self.logger.info(f"  BLEU: {metrics['privacy_metrics']['bleu']:.4f}")
        self.logger.info(f"  ROUGE: {metrics['privacy_metrics']['rouge']:.4f}")
        self.logger.info("\nUtility Preservation (higher is better):")
        self.logger.info(f"  BLEU: {metrics['utility_metrics']['bleu']:.4f}")
        self.logger.info(f"  ROUGE: {metrics['utility_metrics']['rouge']:.4f}")

def main():
    # Get model paths
    with open("outputs/latest_model.txt", "r") as f:
        assistant_model_path = f.read().strip()
    
    # Setup base model
    downloader = ModelDownloader(r"/home/hice1/pli319/scratch/CSE8803/models")
    base_model_path, _ = downloader.download_alpaca()
    
    # Initialize evaluator
    evaluator = UnlearningEvaluator(
        base_model_path=base_model_path,
        assistant_model_path=assistant_model_path,
        alpha=0.75,  # Logit subtraction weight
        filter_threshold=1e-2  # Logit filtering threshold
    )
    
    # Load test data
    test_data = load_data("data/Privacy Violation_test.csv")
    
    # Run evaluation
    metrics = evaluator.evaluate_dataset(test_data, batch_size=4)
    
    # Save results
    evaluator.save_results(metrics)

if __name__ == "__main__":
    main()