# src/trainer.py
import torch
import torch.nn.functional as F
import bitsandbytes as bnb
from typing import List, Dict
from .config import TrainingConfig
import logging
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class UnlearningTrainer:
    def __init__(self, assistant_model, config: TrainingConfig):
        self.model = assistant_model
        self.config = config
        self.device = torch.device(config.device)
        self.best_loss = float('inf')
        
        # Move model to device and ensure float16
        self.model.to(self.device)
        for p in self.model.parameters():
            if p.dtype == torch.float32:
                p.data = p.data.to(torch.float16)
        
        # Configure gradient checkpointing
        self.model.model.gradient_checkpointing_enable()
        
        # Use AdamW with better defaults
        self.optimizer = bnb.optim.AdamW8bit(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)  # Slightly higher beta2
        )

        # # Wrap model in DataParallel
        # if torch.cuda.device_count() > 1:
        #     print(f"Using {torch.cuda.device_count()} GPUs!")
        #     self.model = torch.nn.DataParallel(self.model)

        self.start_epoch = 0  # Track starting epoch for resume
    
    def get_latest_checkpoint(self, checkpoint_dir: str) -> str:
        """Find the latest checkpoint in the directory."""
        if not os.path.exists(checkpoint_dir):
            return None
            
        checkpoints = [f for f in os.listdir(checkpoint_dir) 
                      if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
        
        if not checkpoints:
            return None
            
        # Extract epoch numbers and find the latest
        epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoints]
        latest_epoch = max(epochs)
        
        return os.path.join(checkpoint_dir, f'checkpoint_epoch_{latest_epoch}.pt')
    
    def save_checkpoint(self, epoch: int, loss: float, checkpoint_dir: str, is_best: bool = False):
        """Save a checkpoint of the model."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint for epoch {epoch} at {checkpoint_path}")
        
        # Save best checkpoint if applicable
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with loss {loss:.4f} at {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint of the model."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        return checkpoint['epoch'], checkpoint['loss']
    
    def train(self, train_dataset: List[Dict], checkpoint_dir: str = "checkpoints"):
        self.model.train()
        logger.info(f"Starting training with {len(train_dataset)} examples")
        
        # Create checkpoint directory
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to resume from latest checkpoint
        latest_checkpoint = self.get_latest_checkpoint(str(checkpoint_dir))
        if latest_checkpoint:
            try:
                self.start_epoch, prev_loss = self.load_checkpoint(latest_checkpoint)
                self.best_loss = min(self.best_loss, prev_loss)
                logger.info(f"Resumed from epoch {self.start_epoch} with loss {prev_loss:.4f}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {str(e)}. Starting from scratch.")
                self.start_epoch = 0
        else:
            logger.info("No checkpoint found. Starting from scratch.")
        
        for epoch in range(self.start_epoch, self.start_epoch + self.config.num_epochs):
            epoch_losses = {
                'total': [], 'forget': [], 'retain': []
            }
            
            pbar = tqdm(
                range(0, len(train_dataset), self.config.batch_size),
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs + self.start_epoch}",
                dynamic_ncols=True
            )
            
            for i in pbar:
                batch_end = min(i + self.config.batch_size, len(train_dataset))
                batch = train_dataset[i:batch_end]
                
                try:
                    batch_dict = {
                        "input_ids": torch.cat([item["input_ids"] for item in batch]).to(self.device),
                        "attention_mask": torch.cat([item["attention_mask"] for item in batch]).to(self.device),
                        "labels": torch.cat([item["labels"] for item in batch]).to(self.device)
                    }

                    with torch.set_grad_enabled(True):
                        outputs = self.model(**batch_dict)
                        forget_loss = outputs.loss
                        
                        logits = outputs.logits.view(-1, outputs.logits.size(-1))
                        uniform_target = torch.ones_like(logits) / logits.size(-1)
                        retain_loss = -F.kl_div(
                            F.log_softmax(logits, dim=-1),
                            uniform_target,
                            reduction='batchmean'
                        )
                        
                        # if isinstance(self.model, torch.nn.DataParallel):
                        #     forget_loss = forget_loss.mean()  # Average across GPUs
                        #     retain_loss = retain_loss.mean()

                        loss = forget_loss - self.config.retain_weight * retain_loss
                    
                    epoch_losses['total'].append(loss.item())
                    epoch_losses['forget'].append(forget_loss.item())
                    epoch_losses['retain'].append(retain_loss.item())
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'forget_loss': f"{forget_loss.item():.4f}",
                        'retain_loss': f"{retain_loss.item():.4f}"
                    })
                    
                except Exception as e:
                    logger.error(f"Error in batch {i}: {str(e)}")
                    raise
            
            # Calculate average losses for the epoch
            avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
            
            # Save checkpoint after each epoch
            self.save_checkpoint(
                epoch + 1,
                avg_losses['total'],
                checkpoint_dir,
                is_best=avg_losses['total'] < self.best_loss
            )
            
            # Update best loss if needed
            if avg_losses['total'] < self.best_loss:
                self.best_loss = avg_losses['total']
            
            logger.info(
                f"Epoch {epoch + 1} completed. "
                f"Avg total loss: {avg_losses['total']:.4f}, "
                f"Avg forget loss: {avg_losses['forget']:.4f}, "
                f"Avg retain loss: {avg_losses['retain']:.4f}"
            )