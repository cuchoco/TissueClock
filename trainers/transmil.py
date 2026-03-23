import os
from pathlib import Path
from typing import Optional, Tuple
import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator
from omegaconf import DictConfig
import hydra
import tqdm
import wandb

from dataset.data import AGE_MEAN, AGE_STD, get_abmil_dataloader
from model.transmil import TissueTransMIL


class TransMILTrainer:
    """
    TransMIL Trainer
    using Accelerate for distributed training
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
        # Create output directory
        self.output_dir = Path(cfg.output_dir) / f"tissue_transmil_fold{cfg.fold}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config
        self.fold = cfg.fold
        self.batch_size = cfg.get('batch_size', 1)
        self.num_epochs = cfg.get('num_epochs', 10)
        self.learning_rate = cfg.get('learning_rate', 1e-4)
        self.weight_decay = cfg.get('weight_decay', 1e-5)
        self.accumulation_steps = cfg.get('accumulation_steps', 8)
        self.num_tissues = cfg.get('num_tissues', 40)
        
        # Model parameters
        model_params = cfg.get('model_params', {})
        self.in_dim = model_params.get('in_dim', 1536)
        self.dim = model_params.get('dim', 512)
        self.tissue_embed = model_params.get('tissue_embed', False)
        self.tissue_embed_dim = model_params.get('tissue_embed_dim', 16)
        self.nystrom_dim_head = model_params.get('nystrom_dim_head', self.dim // 8)
        self.nystrom_heads = model_params.get('nystrom_heads', 8)
        self.nystrom_num_landmarks = model_params.get('nystrom_num_landmarks', self.dim // 2)
        self.nystrom_pinv_iterations = model_params.get('nystrom_pinv_iterations', 6)
        self.nystrom_residual = model_params.get('nystrom_residual', True)
        self.ppeg_kernel_sizes = model_params.get('ppeg_kernel_sizes', [7, 5, 3])
        
        # W&B configuration
        self.use_wandb = cfg.get('use_wandb', True)
        self.wandb_project = cfg.get('wandb_project', 'age-predict')
        self.wandb_entity = cfg.get('wandb_entity', None)
        self.wandb_run = None
        self.log_step = cfg.get('log_step', 100)
        
        # Initialize model, optimizer, scheduler
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loss_fn = nn.MSELoss()
        
        # History for tracking (used for checkpointing best model)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_r2': [],
            'best_val_idx': 0
        }
        # Initialize Accelerator
        self.accelerator = Accelerator(gradient_accumulation_steps=self.accumulation_steps)        

        # Initialize W&B
        if self.use_wandb and self.accelerator.is_main_process:
            self._init_wandb()
            
    def _init_wandb(self):
        """Initialize Weights & Biases"""
        try:
            self.wandb_run = wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                config=dict(self.cfg),
                name=f"{self.cfg.model}_fold{self.fold}_label_{self.tissue_embed}",
                dir=str(self.output_dir),
                reinit=False
            )
            self.accelerator.print(f"Initialized W&B run: {self.wandb_run.url}")
        except Exception as e:
            self.accelerator.print(f"Failed to initialize W&B: {e}")
            self.use_wandb = False
        
    def _build_model(self):
        """Build model"""
        self.accelerator.print(f"Building TissueTransMIL model...")
        self.accelerator.print(f"  num_tissues: {self.num_tissues}")
        self.accelerator.print(f"  in_dim: {self.in_dim}")
        self.accelerator.print(f"  dim: {self.dim}")
        self.accelerator.print(f"  tissue_embed: {self.tissue_embed}")
        
        self.model = TissueTransMIL(
            num_tissues=self.num_tissues,
            in_dim=self.in_dim,
            dim=self.dim,
            tissue_embed=self.tissue_embed,
            tissue_embed_dim=self.tissue_embed_dim if self.tissue_embed else 0,
            nystrom_dim_head=self.nystrom_dim_head,
            nystrom_heads=self.nystrom_heads,
            nystrom_num_landmarks=self.nystrom_num_landmarks,
            nystrom_pinv_iterations=self.nystrom_pinv_iterations,
            nystrom_residual=self.nystrom_residual,
            ppeg_kernel_sizes=self.ppeg_kernel_sizes
        )
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm.tqdm(
            train_loader, 
            desc="Training",
            disable=not self.accelerator.is_main_process,
            dynamic_ncols=True
        )
        
        for batch_idx, data in enumerate(progress_bar):
            # Forward pass
            features = data[0]
            ages = data[1]
            tissue_ids = data[2]
            attn_mask = data[3]

            with self.accelerator.accumulate(self.model):
                if self.tissue_embed:
                    predictions, _ = self.model(features, attn_mask=attn_mask, tissue_id=tissue_ids)
                else:
                    predictions, _ = self.model(features, attn_mask=attn_mask, tissue_id=None)
                
                loss = self.train_loss_fn(predictions, ages)
                
                # Backward pass
                self.accelerator.backward(loss)
                # Optimizer and Scheduler step
                self.optimizer.step()
                self.scheduler.step()  
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log to W&B
            if self.use_wandb and self.accelerator.is_main_process and ((batch_idx + 1) % self.log_step) == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/learning_rate_step': current_lr
                })
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader) -> Tuple[float, float, float]:
        """Validate model"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        progress_bar = tqdm.tqdm(
            val_loader,
            desc="Validating",
            disable=not self.accelerator.is_main_process,
            dynamic_ncols=True
        )
        
        for data in progress_bar:
            features = data[0]
            ages = data[1]
            tissue_ids = data[2]
            attn_mask = data[3]
            
            if self.tissue_embed:
                predictions, _ = self.model(features, attn_mask=attn_mask, tissue_id=tissue_ids)
            else:
                predictions, _ = self.model(features, attn_mask=attn_mask, tissue_id=None)
            
            loss = self.train_loss_fn(predictions, ages)
            
            # Gather predictions and targets from all processes
            predictions_gathered = self.accelerator.gather_for_metrics(predictions)
            ages_gathered = self.accelerator.gather_for_metrics(ages)
            
            all_predictions.append(predictions_gathered.cpu().numpy())
            all_targets.append(ages_gathered.cpu().numpy())
            total_loss += loss.item()
        
        # Concatenate all predictions and targets
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Denormalize for metric calculation
        all_predictions_denorm = all_predictions * AGE_STD + AGE_MEAN
        all_targets_denorm = all_targets * AGE_STD + AGE_MEAN
        
        # Calculate metrics
        val_loss = total_loss / len(val_loader)
        mae = mean_absolute_error(all_targets_denorm, all_predictions_denorm)
        r2 = r2_score(all_targets_denorm, all_predictions_denorm)
        
        # Log to W&B
        if self.use_wandb and self.accelerator.is_main_process:
            wandb.log({
                'val/loss': val_loss,
                'val/mae': mae,
                'val/r2': r2
            })
        
        return val_loss, mae, r2
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.cfg
        }
        
        self.accelerator.save(save_dict, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            self.accelerator.save(save_dict, best_path)
            self.accelerator.print(f"Saved best model checkpoint to {best_path}")
            
            # Save best model to W&B
            if self.use_wandb and self.accelerator.is_main_process:
                wandb.save(str(best_path), base_path=str(self.checkpoint_dir))
        
        self.accelerator.print(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        
        self.accelerator.print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint.get('epoch', 0)
    
    def train(self):
        """Main training loop"""
        self.accelerator.print("=" * 80)
        self.accelerator.print(f"Starting TransMIL Training (Fold {self.fold})")
        self.accelerator.print("=" * 80)
        
        # Model build
        self._build_model()
        
        # Data loader (reusing ABMIL dataloader since they have same requirements)
        if self.cfg.get('use_normal_data', False):
            from dataset.data_normal import get_abmil_dataloader as get_abmil_dataloader_normal
            self.accelerator.print("Using Normal Data ONLY!")
            train_loader, val_loader = get_abmil_dataloader_normal(fold=self.fold, batch_size=self.batch_size)
        else:
            train_loader, val_loader = get_abmil_dataloader(fold=self.fold, batch_size=self.batch_size)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Accelerator prepare
        self.model, self.optimizer, train_loader, val_loader = self.accelerator.prepare(
            self.model, self.optimizer, train_loader, val_loader
        )
        
        steps_per_epoch = math.ceil(len(train_loader) / self.accumulation_steps)
        total_steps = steps_per_epoch * self.num_epochs
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps
        )
        self.scheduler = self.accelerator.prepare(self.scheduler)
        
        self.accelerator.print(f"Optimizer: AdamW (lr={self.learning_rate}, weight_decay={self.weight_decay})")
        self.accelerator.print(f"Scheduler: CosineAnnealingLR (T_max={total_steps} steps)")
        
        # Training loop
        best_val_mae = float('inf')
        
        self.accelerator.print(f"Training for {self.num_epochs} epochs")
        self.accelerator.print(f"Training batches per epoch: {len(train_loader)}")
        self.accelerator.print(f"Validation batches per epoch: {len(val_loader)}")
        
        for epoch in range(self.num_epochs):
            self.accelerator.print(f"\nEpoch [{epoch+1}/{self.num_epochs}]")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss, val_mae, val_r2 = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)
            self.history['val_r2'].append(val_r2)
            
            # Log results
            self.accelerator.print(
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val MAE: {val_mae:.4f} | "
                f"Val R²: {val_r2:.4f}"
            )
            
            # Log epoch metrics to W&B
            if self.use_wandb and self.accelerator.is_main_process:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_loss,
                })
            
            # Save checkpoint
            new_best = val_mae < best_val_mae
            if new_best:
                best_val_mae = val_mae
                self.history['best_val_idx'] = epoch
                self.accelerator.print(f"New best model! VAL MAE: {val_mae:.4f}")
            
            self.save_checkpoint(epoch, is_best=new_best)
        
        self.accelerator.print("\n" + "=" * 80)
        self.accelerator.print("Training completed!")
        self.accelerator.print(f"Best model at epoch {self.history['best_val_idx']}")
        self.accelerator.print(f"Best VAL MAE: {best_val_mae:.4f}")
        self.accelerator.print("=" * 80)
        
        # Finish W&B run
        if self.use_wandb and self.accelerator.is_main_process:
            wandb.summary['best_epoch'] = self.history['best_val_idx']
            wandb.summary['best_val_mae'] = best_val_mae
            wandb.finish()
        
        return self.history


def train(cfg):
    """Train function for TransMIL model (compatible with TRAINERS registry)"""
    trainer = TransMILTrainer(cfg)
    history = trainer.train()
    return history


@hydra.main(version_base=None, config_path="config", config_name="config_transmil")
def main(cfg: DictConfig):
    """Main training function for direct script execution"""
    train(cfg)


if __name__ == "__main__":
    main()
