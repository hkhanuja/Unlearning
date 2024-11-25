import warnings

# Filter warnings
warnings.filterwarnings("ignore", message=".*_register_pytree_node*")
warnings.filterwarnings("ignore", message=".*MatMul8bitLt: inputs will be cast.*")
warnings.filterwarnings("ignore", message=".*gradient checkpointing.*")
warnings.filterwarnings("ignore", message=".*torch.load*")

# run.py
import os
from datetime import datetime
from src.config import ModelConfig, LoRAConfig, TrainingConfig
from src.unlearning import AlpacaUnlearning
from utils.data_utils import load_data, save_model
from utils.model_utils import ModelDownloader
from utils.logger import setup_logger
import logging
import torch

# Set logging level for transformers
logging.getLogger("transformers").setLevel(logging.ERROR)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def get_or_create_checkpoint_dir(logger):
    """Get the latest checkpoint directory or create a new one."""
    base_checkpoint_dir = r"/home/hice1/pli319/scratch/CSE8803/checkpoints"
    if not os.path.exists(base_checkpoint_dir):
        new_dir = os.path.join(base_checkpoint_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(new_dir)
        return new_dir
        
    # Check for existing checkpoint directories
    dirs = [d for d in os.listdir(base_checkpoint_dir) 
            if os.path.isdir(os.path.join(base_checkpoint_dir, d))]
    
    if not dirs:
        new_dir = os.path.join(base_checkpoint_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(new_dir)
        return new_dir
        
    # Return the most recent directory
    latest_dir = max(dirs, key=lambda x: os.path.getmtime(os.path.join(base_checkpoint_dir, x)))
    latest_path = os.path.join(base_checkpoint_dir, latest_dir)
    
    logger.info(f"Found existing checkpoint directory: {latest_path}")
    return latest_path

def main():
    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = r"/home/hice1/pli319/scratch/CSE8803/logs"
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(
        "training",
        log_file=f"{log_dir}/training_{timestamp}.log"
    )
    
    try:
        # Get or create checkpoint directory
        checkpoint_dir = get_or_create_checkpoint_dir(logger)
        logger.info(f"Using checkpoint directory: {checkpoint_dir}")

        # Download model
        logger.info("Downloading Alpaca model...")
        downloader = ModelDownloader(r"/home/hice1/pli319/scratch/CSE8803/models")
        model_path, tokenizer_path = downloader.download_alpaca()
        
        # Create output directory
        output_dir = r"/home/hice1/pli319/scratch/CSE8803/outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize configurations
        logger.info("Initializing configurations...")
        model_config = ModelConfig(
            model_path=model_path,
            num_assistant_layers=8,
            load_in_8bit=False
        )
        
        lora_config = LoRAConfig(
            r=64,
            alpha=64,
            dropout=0.1
        )
        
        training_config = TrainingConfig(
            learning_rate=5e-4,
            num_epochs=50,
            batch_size=4
        )
        
        logger.info("Initializing unlearning system...")
        unlearner = AlpacaUnlearning(
            model_config=model_config,
            lora_config=lora_config,
            training_config=training_config
        )
        
        logger.info("Loading training data...")
        train_data = load_data("data/Privacy Violation_train.csv")
        logger.info(f"Loaded {len(train_data)} training examples")
        logger.info(f"Checkpoints will be saved to {checkpoint_dir}")
        
        # Train with checkpointing
        logger.info("Starting training...")
        unlearner.train(train_data, checkpoint_dir=checkpoint_dir)
        
        # Save final model
        final_model_path = os.path.join("outputs", f"unlearned_model_{timestamp}")
        logger.info(f"Saving final model to {final_model_path}")
        save_model(unlearner, final_model_path)
        
        # Save best model path for evaluation
        best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
        with open(os.path.join("outputs", "latest_model.txt"), "w") as f:
            f.write(best_model_path)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()