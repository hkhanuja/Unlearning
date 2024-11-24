import warnings

# Filter warnings
warnings.filterwarnings("ignore", message=".*_register_pytree_node*")
warnings.filterwarnings("ignore", message=".*MatMul8bitLt: inputs will be cast.*")
warnings.filterwarnings("ignore", message=".*gradient checkpointing.*")
warnings.filterwarnings("ignore", message=".*torch.load*")

import os
import torch
from tqdm import tqdm
from datetime import datetime
from src.config import ModelConfig, LoRAConfig, TrainingConfig
from src.unlearning import AlpacaUnlearning
from utils.data_utils import load_data, load_model
from utils.model_utils import ModelDownloader
from utils.logger import setup_logger

def evaluate_with_logit_subtraction(original_model, assistant_model, test_data, logger, 
                                 alpha=0.75, filter_threshold=1e-2, max_length=512):
   original_responses = []
   unlearned_responses = []
   ground_truth = []
   
   logger.info("Generating responses with logit subtraction...")
   
   for item in tqdm(test_data):
       # Get logits from both models
       inputs = original_model.tokenizer(
           item["prompt"],
           return_tensors="pt",
           max_length=max_length,
           truncation=True
       ).to(original_model.base_model.device)
       
       with torch.no_grad():
           base_outputs = original_model.base_model.generate(
               **inputs,
               max_length=max_length,
               output_scores=True,
               return_dict_in_generate=True,
           )
           
           assistant_outputs = assistant_model.generate(
               **inputs,
               max_length=max_length,
               output_scores=True,
               return_dict_in_generate=True,
           )
       
       # Perform logit subtraction with filtering
       filtered_scores = []
       for base_score, assistant_score in zip(base_outputs.scores, assistant_outputs.scores):
           mask = torch.abs(assistant_score) > filter_threshold
           filtered_assistant = assistant_score * mask
           final_score = base_score - alpha * filtered_assistant
           filtered_scores.append(final_score)
       
       # Generate tokens from modified logits
       final_tokens = torch.cat([
           torch.argmax(score, dim=-1) for score in filtered_scores
       ])
       
       unlearned_response = original_model.tokenizer.decode(final_tokens, skip_special_tokens=True)
       original_response = original_model.tokenizer.decode(
           base_outputs.sequences[0], skip_special_tokens=True
       )
       
       original_responses.append(original_response)
       unlearned_responses.append(unlearned_response)
       ground_truth.append(item["response"])
       
       # Clear cache
       torch.cuda.empty_cache()
   
   return original_responses, unlearned_responses, ground_truth

def main():
   timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
   logger = setup_logger("evaluation", fr"/home/hice1/pli319/scratch/CSE8803/log/evaluation_{timestamp}.log")
   
   try:
       # Load model paths and configs
       with open("outputs/latest_model.txt", "r") as f:
           unlearned_model_path = f.read().strip()
           
       downloader = ModelDownloader(r"/home/hice1/pli319/scratch/CSE8803/models")
       model_path, _ = downloader.download_alpaca()
       
       model_config = ModelConfig(
           model_path=model_path,
           num_assistant_layers=8,
           load_in_8bit=False
       )
       
       # Initialize models
       logger.info("Loading models...")
       original_model = AlpacaUnlearning(model_config=model_config)
       assistant_model = AlpacaUnlearning(model_config=model_config)
       load_model(assistant_model, unlearned_model_path)
       
       # Load test data
       test_data = load_data("data/Privacy Violation_test.csv")
       
       # Generate responses
       original_responses, unlearned_responses, ground_truth = evaluate_with_logit_subtraction(
           original_model, assistant_model, test_data, logger
       )
       
       # Calculate metrics and save results
       metrics = calculate_metrics(
           original_responses,
           unlearned_responses, 
           ground_truth,
           logger
       )
       
       # Save results
       save_results(metrics, original_responses, unlearned_responses, ground_truth, timestamp)
       
   except Exception as e:
       logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
       raise

if __name__ == "__main__":
   main()