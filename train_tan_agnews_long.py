#!/usr/bin/env python3
"""
Train TAN/Topoformer on AG News Dataset - Multi-GPU with DistributedDataParallel
Usage: CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_tan_agnews_long.py

Author: TAN Research Team
Date: 2024
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
import time
from datetime import datetime
import gc
import random

# Metrics
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Transformers and datasets
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Import TAN/Topoformer components
try:
    from streamlined_topoformer import TopoformerConfig, TopoformerLayer
    from tan_summarization_dataparallel import TopoformerForSummarization
except ImportError:
    raise ImportError("Topoformer modules not found. Please ensure required files are available.")

def setup_logging(rank):
    """Setup logging for distributed training"""
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('agnews_training_ddp.log'),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.WARNING)
    
    return logging.getLogger(__name__)

def setup_distributed():
    """Initialize distributed training"""
    # Get rank and world_size from environment variables set by torchrun
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # Initialize process group
    dist.init_process_group(backend='nccl')
    
    return rank, local_rank, world_size, device

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

@dataclass
class AGNewsConfig:
    """Configuration for AG News training"""
    # Data parameters
    data_dir: str = './agnews_data'
    save_dir: str = './agnews_models_ddp'
    cache_dir: str = './cache'
    max_samples: int = 90000  # Limit to 90k samples for consistency
    
    # Model parameters
    vocab_size: int = 50268
    embed_dim: int = 512  # Reduced from 768 to save memory
    num_layers: int = 6   # Reduced from 8 to save memory
    num_heads: int = 8    # Reduced from 12 to save memory
    dropout: float = 0.1
    k_neighbors: int = 32
    use_topology: bool = True
    
    # Sequence parameters
    max_seq_len: int = 512
    
    # Training parameters - adjusted for multi-GPU
    batch_size: int = 8   # Per GPU batch size - total will be 8*2=16
    gradient_accumulation_steps: int = 4  # Increased to maintain effective batch size
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_epochs: int = 7
    max_grad_norm: float = 1.0
    
    # Classification
    num_labels: int = 4  # AG News has 4 classes
    
    # Optimization
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True  # Enable to save memory
    
    # Hardware
    num_workers: int = 2  # Reduced for distributed training
    pin_memory: bool = True
    prefetch_factor: int = 2


class TopoformerForClassification(nn.Module):
    """TAN/Topoformer adapted for text classification with memory optimization"""
    
    def __init__(self, config: TopoformerConfig, num_labels: int, use_gradient_checkpointing: bool = False):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.embedding_norm = nn.LayerNorm(config.embed_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        
        # Topoformer layers
        self.layers = nn.ModuleList([
            TopoformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Pooling and classification head
        self.pooler = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.Tanh(),
            nn.Dropout(config.dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim // 2, num_labels)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for classification"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position ids
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        embeddings = token_embeds + position_embeds
        embeddings = self.embedding_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        
        # Apply Topoformer layers with optional gradient checkpointing
        hidden_states = embeddings
        for layer in self.layers:
            if self.use_gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    attention_mask,
                    use_reentrant=False
                )
            else:
                hidden_states = layer(hidden_states, attention_mask)
        
        # Global pooling with attention masking
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (hidden_states * mask_expanded).sum(1)
            sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = hidden_states.mean(1)
        
        pooled = self.pooler(pooled)
        
        # Classification
        logits = self.classifier(pooled)
        
        outputs = {'logits': logits}
        
        # Calculate loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            outputs['loss'] = loss
        
        return outputs


class AGNewsDataset(Dataset):
    """Dataset class for AG News without artificial extension"""
    
    def __init__(
        self,
        split: str,
        tokenizer,
        config: AGNewsConfig,
        max_samples: Optional[int] = None
    ):
        """Initialize dataset with original AG News data"""
        self.split = split
        self.tokenizer = tokenizer
        self.config = config
        
        # Load dataset
        self.dataset = load_dataset(
            'fancyzhx/ag_news',
            split=split,
            cache_dir=config.cache_dir,
            trust_remote_code=True
        )
        
        # Limit samples if specified
        if max_samples is not None and len(self.dataset) > max_samples:
            # Use a deterministic subset for reproducibility
            indices = list(range(0, len(self.dataset), len(self.dataset) // max_samples))[:max_samples]
            self.dataset = self.dataset.select(indices)
        
        # Class labels mapping
        self.label_names = ['World', 'Sports', 'Business', 'Sci/Tech']
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        item = self.dataset[idx]
        
        # Get text and label
        text = item['text'].strip()
        label = item['label']
        
        # Tokenize with proper truncation and padding
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_seq_len,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
            'text': text
        }


class DistributedTrainer:
    """Trainer for distributed multi-GPU training"""
    
    def __init__(
        self,
        model: nn.Module,
        config: AGNewsConfig,
        device: torch.device,
        rank: int,
        world_size: int
    ):
        self.config = config
        self.device = device
        self.rank = rank
        self.world_size = world_size
        
        # Move model to device and wrap with DDP
        self.model = model.to(device)
        self.model = DDP(self.model, device_ids=[device.index])
        
        # Setup logger
        self.logger = setup_logging(rank)
        
        # Optimizer with weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'norm']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': config.weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
        
        # Mixed precision training
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Metrics tracking (only on rank 0)
        if rank == 0:
            self.train_history = []
            self.val_history = []
            self.best_val_acc = 0.0
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        scheduler=None
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        # Only show progress bar on rank 0
        if self.rank == 0:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} - Training")
        else:
            progress_bar = train_loader
        
        accumulation_counter = 0
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # Mixed precision training
            if self.config.use_mixed_precision:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs['loss'] / self.config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs['loss'] / self.config.gradient_accumulation_steps
                loss.backward()
            
            accumulation_counter += 1
            
            # Optimizer step
            if accumulation_counter % self.config.gradient_accumulation_steps == 0:
                if self.config.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                if scheduler is not None:
                    scheduler.step()
                
                accumulation_counter = 0
            
            # Track metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            with torch.no_grad():
                preds = torch.argmax(outputs['logits'], dim=-1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
            
            # Update progress bar only on rank 0
            if self.rank == 0 and isinstance(progress_bar, tqdm):
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * self.config.gradient_accumulation_steps:.4f}',
                    'acc': f'{total_correct / total_samples:.4f}'
                })
            
            # Clear cache periodically
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        # Gather metrics from all processes
        total_loss_tensor = torch.tensor(total_loss, device=self.device)
        total_correct_tensor = torch.tensor(total_correct, device=self.device)
        total_samples_tensor = torch.tensor(total_samples, device=self.device)
        
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        
        # Calculate epoch metrics
        metrics = {
            'loss': total_loss_tensor.item() / (len(train_loader) * self.world_size),
            'accuracy': total_correct_tensor.item() / total_samples_tensor.item()
        }
        
        return metrics
    
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_iter = tqdm(val_loader, desc=f"Epoch {epoch} - Validation") if self.rank == 0 else val_loader
            
            for batch in val_iter:
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs['loss'].item()
                
                preds = torch.argmax(outputs['logits'], dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Gather validation results from all processes
        # Convert to tensors for gathering
        all_preds_tensor = torch.tensor(all_preds, dtype=torch.long, device=self.device)
        all_labels_tensor = torch.tensor(all_labels, dtype=torch.long, device=self.device)
        total_loss_tensor = torch.tensor(total_loss, device=self.device)
        
        # Gather from all processes
        gathered_preds = [torch.zeros_like(all_preds_tensor) for _ in range(self.world_size)]
        gathered_labels = [torch.zeros_like(all_labels_tensor) for _ in range(self.world_size)]
        
        dist.all_gather(gathered_preds, all_preds_tensor)
        dist.all_gather(gathered_labels, all_labels_tensor)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        
        if self.rank == 0:
            # Combine results from all processes
            final_preds = torch.cat(gathered_preds).cpu().numpy()
            final_labels = torch.cat(gathered_labels).cpu().numpy()
            
            # Calculate metrics
            accuracy = accuracy_score(final_labels, final_preds)
            precision, recall, f1_macro, _ = precision_recall_fscore_support(
                final_labels, final_preds, average='macro', zero_division=0
            )
            f1_micro = f1_score(final_labels, final_preds, average='micro')
            
            metrics = {
                'loss': total_loss_tensor.item() / (len(val_loader) * self.world_size),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_macro': f1_macro,
                'f1_micro': f1_micro
            }
        else:
            metrics = {}
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: str
    ):
        """Full training loop"""
        if self.rank == 0:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True, parents=True)
        
        # Setup scheduler
        total_steps = len(train_loader) * num_epochs // self.config.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.config.warmup_ratio * total_steps),
            num_training_steps=total_steps
        )
        
        if self.rank == 0:
            self.logger.info(f"Starting training for {num_epochs} epochs")
            self.logger.info(f"Total training steps: {total_steps}")
            self.logger.info(f"Using {self.world_size} GPUs")
        
        for epoch in range(1, num_epochs + 1):
            # Set epoch for distributed sampler
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            if self.rank == 0:
                self.logger.info(f"\n{'='*50}")
                self.logger.info(f"Epoch {epoch}/{num_epochs}")
                self.logger.info(f"{'='*50}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch, scheduler)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            
            if self.rank == 0:
                self.train_history.append(train_metrics)
                self.val_history.append(val_metrics)
                
                # Log metrics
                self.logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                               f"Accuracy: {train_metrics['accuracy']:.4f}")
                self.logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                               f"Accuracy: {val_metrics['accuracy']:.4f}, "
                               f"F1-Macro: {val_metrics['f1_macro']:.4f}")
                
                # Save best model
                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    self.save_model(save_path / 'best_model.pt', epoch, val_metrics)
                    self.logger.info(f"Saved best model with accuracy: {self.best_val_acc:.4f}")
                
                # Save checkpoint
                if epoch % 3 == 0:
                    self.save_model(save_path / f'checkpoint_epoch_{epoch}.pt', epoch, val_metrics)
            
            # Synchronize all processes
            dist.barrier()
        
        if self.rank == 0:
            # Save final model
            self.save_model(save_path / 'final_model.pt', num_epochs, val_metrics)
            
            # Save training history
            history = {
                'train': self.train_history,
                'val': self.val_history,
                'config': self.config.__dict__
            }
            with open(save_path / 'training_history.json', 'w') as f:
                json.dump(history, f, indent=2)
            
            self.logger.info(f"\nTraining complete! Best validation accuracy: {self.best_val_acc:.4f}")
    
    def save_model(self, path: Path, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        if self.rank == 0:  # Only save on rank 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': metrics,
                'config': self.config.__dict__,
                'best_val_acc': self.best_val_acc if hasattr(self, 'best_val_acc') else 0.0
            }
            
            torch.save(checkpoint, path)
            self.logger.info(f"Model saved to {path}")


def main():
    """Main training script"""
    try:
        # Setup distributed training
        rank, local_rank, world_size, device = setup_distributed()
        
        # Configuration
        config = AGNewsConfig()
        
        # Setup logger
        logger = setup_logging(rank)
        
        if rank == 0:
            logger.info(f"Using device: {device}")
            logger.info(f"World size: {world_size}")
            logger.info(f"Local rank: {local_rank}")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        config.vocab_size = tokenizer.vocab_size
        
        # Create datasets
        if rank == 0:
            logger.info("\n" + "="*60)
            logger.info("Loading AG News Dataset")
            logger.info("="*60)
        
        # Load train dataset with sample limit
        full_train_dataset = AGNewsDataset('train', tokenizer, config, max_samples=config.max_samples)
        
        # Create train/val split (90/10)
        train_size = int(0.9 * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            sampler=val_sampler,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False
        )
        
        if rank == 0:
            logger.info(f"Train samples: {len(train_dataset)}")
            logger.info(f"Val samples: {len(val_dataset)}")
            logger.info(f"Train batches per GPU: {len(train_loader)}")
            logger.info(f"Val batches per GPU: {len(val_loader)}")
        
        # Create model
        if rank == 0:
            logger.info("\n" + "="*60)
            logger.info("Initializing TAN Model for Classification")
            logger.info("="*60)
        
        # Create TopoformerConfig without gradient_checkpointing parameter
        topo_config = TopoformerConfig(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            k_neighbors=config.k_neighbors,
            use_topology=config.use_topology
        )
        
        # Pass gradient_checkpointing as a separate parameter to the model
        model = TopoformerForClassification(
            topo_config, 
            config.num_labels,
            use_gradient_checkpointing=config.gradient_checkpointing
        )
        
        if rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Initialize trainer
        trainer = DistributedTrainer(
            model=model,
            config=config,
            device=device,
            rank=rank,
            world_size=world_size
        )
        
        # Train model
        if rank == 0:
            logger.info("\n" + "="*60)
            logger.info("Starting Distributed Training")
            logger.info("="*60)
        
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.num_epochs,
            save_dir=config.save_dir
        )
        
        if rank == 0:
            logger.info("\n" + "="*60)
            logger.info("Training Complete!")
            logger.info("="*60)
            logger.info(f"Results saved to: {config.save_dir}")
    
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
