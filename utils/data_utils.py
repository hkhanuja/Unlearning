# utils/data_utils.py
import pandas as pd
from typing import Tuple, List, Dict
import os
import logging

logger = logging.getLogger(__name__)

def load_data(file_path: str) -> List[Dict[str, str]]:
    """Load and process privacy violation data from CSV."""
    df = pd.read_csv(file_path)
    
    data = []
    for _, row in df.iterrows():
        data.append({
            "prompt": row["prompt"],
            "response": row["response"]
        })
    
    return data

def save_model(model, save_path: str):
    """Save the unlearned model state."""
    try:
        logger.info(f"Saving model to {save_path}")
        os.makedirs(save_path, exist_ok=True)
        
        # Save assistant model
        model.assistant_model.save_pretrained(os.path.join(save_path, "assistant_model"))
        
        logger.info("Model saved successfully")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_model(model, load_path: str):
    """Load the unlearned model state."""
    try:
        logger.info(f"Loading model from {load_path}")
        
        # Load assistant model
        model.assistant_model.load_pretrained(os.path.join(load_path, "assistant_model"))
        
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise