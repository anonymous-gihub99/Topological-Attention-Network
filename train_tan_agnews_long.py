#!/usr/bin/env python3
"""
Train TAN/Topoformer on AG News Dataset - DataParallel with 2 GPUs
Optimized for memory efficiency and proper GPU utilization
Usage: python train_tan_agnews_dataparallel.py

Author: TAN Research Team
Date: 2024
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, Dataset, random_split
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
import psutil
import GPUtil

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
except ImportError:
    raise ImportError("Topoformer modules not found. Please ensure required files are available.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agnews_training_dataparallel.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_memory_stats():
    """Get current memory statistics for debugging"""
    stats = {}
    
    # CPU Memory
    process = psutil.Process(os.getpid())
    stats['cpu_memory_gb'] = process.memory_info().rss / 1024 / 1024 / 1024
    
    # GPU Memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024 / 1024 / 1024
            reserved = torch.cuda.memory_reserved(i) / 1024 / 1024 / 1024
            stats[f'gpu{i}_allocated_gb'] = allocated
            stats[f'gpu{i}_reserved_gb'] = reserved
        
        # Using GPUtil for more detailed stats
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            stats[f'gpu{i}_used_mb'] = gpu.memoryUsed
            stats[f'gpu{i}_total_mb'] = gpu.memoryTotal
            stats[f'gpu{i}_util_percent'] = gpu.memoryUtil * 100
    
    return stats


def log_memory_usage(phase=""):
    """Log current memory usage"""
    stats = get_memory_stats()
    logger.info(f"Memory Usage {phase}:")
    for key, value in stats.items():
        if 'gb' in key:
            logger.info(f"  {key}: {value:.2f} GB")
        elif 'mb' in key:
            logger.info(f"  {key}: {value:.0f} MB")
        elif 'percent' in key:
            logger.info(f"  {key}: {value:.1f}%")


@dataclass
class AGNewsConfig:
    """Configuration for AG News training with DataParallel"""
    # Data parameters
    data_dir: str = './agnews_data'
    save_dir: str = './agnews_models_dataparallel'
    cache_dir: str = './cache'
    max_samples: int = 90000  # Limit to 90k samples for consistency
    
    # Model parameters - Optimized for 2 GPUs
    vocab_size: int = 50268
    embed_dim: int = 512  # Reduced for memory efficiency
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    k_neighbors: int = 32
    use_topology: bool = True
    
    # Sequence parameters
    max_seq_len: int = 512
    
    # Training parameters - Optimized for DataParallel on 2 GPUs
    batch_size_per_gpu: int = 16  # Per GPU batch size
    total_batch_size: int = 32  # Total batch size (16 * 2 GPUs)
    gradient_accumulation_steps: int = 2  # For effective batch size of 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_epochs: int = 10
    max_grad_norm: float = 1.0
    
    # Classification
    num_labels: int = 4  # AG News has 4 classes
    
    # Optimization
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True  # Enable to save memory
    
    # Hardware - Optimized for DataParallel
    num_gpus: int = 2  # Explicitly set to 2
    num_workers: int = 4  # 2 workers per GPU
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True  # Keep workers alive between epochs
    
    # Memory management
    empty_cache_freq: int = 100  # Clear cache every N batches
    log_memory_freq: int = 200  # Log memory usage every N batches


class TopoformerForClassification(nn.Module):
    """TAN/Topoformer for text classification with memory optimization"""
    
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
    """Dataset class for AG News"""
    
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
        logger.info(f"Loading AG News {split} dataset...")
        self.dataset = load_dataset(
            'fancyzhx/ag_news',
            split=split,
            cache_dir=config.cache_dir,
            trust_remote_code=True
        )
        
        # Limit samples if specified
        if max_samples is not None and len(self.dataset) > max_samples:
            indices = list(range(min(len(self.dataset), max_samples)))
            self.dataset = self.dataset.select(indices)
        
        # Class labels mapping
        self.label_names = ['World', 'Sports', 'Business', 'Sci/Tech']
        
        logger.info(f"Loaded {len(self.dataset)} samples for {split}")
        
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


class DataParallelTrainer:
    """Trainer optimized for DataParallel on 2 GPUs"""
    
    def __init__(
        self,
        model: nn.Module,
        config: AGNewsConfig,
        device: torch.device
    ):
        self.config = config
        self.device = device
        
        # Check available GPUs
        num_gpus = torch.cuda.device_count()
        if num_gpus < 2:
            logger.warning(f"Only {num_gpus} GPU(s) available. Requested 2.")
            self.gpu_ids = list(range(num_gpus))
        else:
            self.gpu_ids = [0, 1]  # Use first 2 GPUs
            logger.info(f"Using GPUs: {self.gpu_ids}")
        
        # Move model to primary device and wrap with DataParallel
        self.model = model.to(device)
        if len(self.gpu_ids) > 1:
            self.model = DataParallel(self.model, device_ids=self.gpu_ids)
            logger.info("Model wrapped with DataParallel")
        
        # Log initial memory state
        log_memory_usage("Initial")
        
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
        
        # Metrics tracking
        self.train_history = []
        self.val_history = []
        self.best_val_acc = 0.0
        
        # Memory management
        self.batch_count = 0
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        scheduler=None
    ) -> Dict[str, float]:
        """Train for one epoch with memory management"""
        self.model.train()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} - Training")
        accumulation_counter = 0
        
        for batch_idx, batch in enumerate(progress_bar):
            self.batch_count += 1
            
            # Move batch to device (primary GPU)
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
                    loss = outputs['loss']
                    
                    # Handle DataParallel loss averaging
                    if isinstance(self.model, DataParallel):
                        loss = loss.mean()
                    
                    loss = loss / self.config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs['loss']
                
                # Handle DataParallel loss averaging
                if isinstance(self.model, DataParallel):
                    loss = loss.mean()
                
                loss = loss / self.config.gradient_accumulation_steps
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
                # Get logits from all GPUs
                if isinstance(self.model, DataParallel):
                    # DataParallel already gathered the outputs
                    logits = outputs['logits']
                else:
                    logits = outputs['logits']
                
                preds = torch.argmax(logits, dim=-1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item() * self.config.gradient_accumulation_steps:.4f}',
                'acc': f'{total_correct / total_samples:.4f}',
                'gpu0_mem': f'{torch.cuda.memory_allocated(0)/1e9:.1f}GB',
                'gpu1_mem': f'{torch.cuda.memory_allocated(1)/1e9:.1f}GB' if len(self.gpu_ids) > 1 else 'N/A'
            })
            
            # Memory management
            if batch_idx % self.config.empty_cache_freq == 0:
                torch.cuda.empty_cache()
                gc.collect()
            
            # Log memory usage periodically
            if batch_idx % self.config.log_memory_freq == 0:
                log_memory_usage(f"Batch {batch_idx}")
        
        # Clear cache at end of epoch
        torch.cuda.empty_cache()
        gc.collect()
        
        # Calculate epoch metrics
        metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': total_correct / total_samples
        }
        
        return metrics
    
    def validate(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Validate model with memory efficiency"""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch} - Validation")):
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                
                if self.config.use_mixed_precision:
                    with autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                
                loss = outputs['loss']
                if isinstance(self.model, DataParallel):
                    loss = loss.mean()
                
                total_loss += loss.item()
                
                preds = torch.argmax(outputs['logits'], dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Clear cache periodically during validation
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        f1_micro = f1_score(all_labels, all_preds, average='micro')
        
        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro
        }
        
        # Clear cache after validation
        torch.cuda.empty_cache()
        gc.collect()
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: str
    ):
        """Full training loop with memory monitoring"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        # Setup scheduler
        total_steps = len(train_loader) * num_epochs // self.config.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.config.warmup_ratio * total_steps),
            num_training_steps=total_steps
        )
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Batch size per GPU: {self.config.batch_size_per_gpu}")
        logger.info(f"Total batch size: {self.config.total_batch_size}")
        logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {self.config.total_batch_size * self.config.gradient_accumulation_steps}")
        
        # Log initial GPU memory
        log_memory_usage("Before Training")
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"{'='*50}")
            
            # Log memory at start of epoch
            log_memory_usage(f"Start of Epoch {epoch}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch, scheduler)
            self.train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            self.val_history.append(val_metrics)
            
            # Log metrics
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Accuracy: {train_metrics['accuracy']:.4f}")
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"Accuracy: {val_metrics['accuracy']:.4f}, "
                       f"F1-Macro: {val_metrics['f1_macro']:.4f}")
            
            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_model(save_path / 'best_model.pt', epoch, val_metrics)
                logger.info(f"Saved best model with accuracy: {self.best_val_acc:.4f}")
            
            # Save checkpoint periodically
            if epoch % 3 == 0:
                self.save_model(save_path / f'checkpoint_epoch_{epoch}.pt', epoch, val_metrics)
            
            # Clear cache and collect garbage at end of epoch
            torch.cuda.empty_cache()
            gc.collect()
            
            # Log memory at end of epoch
            log_memory_usage(f"End of Epoch {epoch}")
        
        # Save final model
        self.save_model(save_path / 'final_model.pt', num_epochs, val_metrics)
        
        # Save training history
        history = {
            'train': self.train_history,
            'val': self.val_history,
            'config': self.config.__dict__
        }
        with open(save_path / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        # Plot training curves
        self.plot_training_curves(save_path)
        
        logger.info(f"\nTraining complete! Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Final memory report
        log_memory_usage("Training Complete")
    
    def save_model(self, path: Path, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        # Extract model from DataParallel wrapper if necessary
        model_to_save = self.model.module if isinstance(self.model, DataParallel) else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__,
            'best_val_acc': self.best_val_acc,
            'gpu_ids': self.gpu_ids
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def plot_training_curves(self, save_dir: Path):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        epochs = range(1, len(self.train_history) + 1)
        
        # Loss
        ax = axes[0, 0]
        ax.plot(epochs, [h['loss'] for h in self.train_history], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, [h['loss'] for h in self.val_history], 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy
        ax = axes[0, 1]
        ax.plot(epochs, [h['accuracy'] for h in self.train_history], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, [h['accuracy'] for h in self.val_history], 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training and Validation Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # F1 Scores
        ax = axes[1, 0]
        ax.plot(epochs, [h['f1_macro'] for h in self.val_history], 'g-', label='F1-Macro', linewidth=2)
        ax.plot(epochs, [h['f1_micro'] for h in self.val_history], 'm-', label='F1-Micro', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title('Validation F1 Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Precision and Recall
        ax = axes[1, 1]
        ax.plot(epochs, [h['precision'] for h in self.val_history], 'c-', label='Precision', linewidth=2)
        ax.plot(epochs, [h['recall'] for h in self.val_history], 'orange', label='Recall', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Validation Precision and Recall')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'AG News Training Curves (DataParallel - {len(self.gpu_ids)} GPUs)', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    save_dir: Path
) -> Dict[str, float]:
    """Test the trained model"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = torch.argmax(outputs['logits'], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    label_names = ['World', 'Sports', 'Business', 'Sci/Tech']
    report = classification_report(all_labels, all_preds, target_names=label_names)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    # Save results
    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix - AG News Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("\nTest Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Macro: {f1_macro:.4f}")
    logger.info(f"F1-Micro: {f1_micro:.4f}")
    logger.info(f"\nClassification Report:\n{report}")
    
    return results


def main():
    """Main training script for DataParallel on 2 GPUs"""
    # Set CUDA settings for optimal performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Configuration
    config = AGNewsConfig()
    
    # Check GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPUs.")
    
    num_gpus = torch.cuda.device_count()
    logger.info(f"Found {num_gpus} GPU(s)")
    
    if num_gpus < 2:
        logger.warning(f"Script is optimized for 2 GPUs but found {num_gpus}. Continuing with available GPUs.")
        config.num_gpus = num_gpus
        config.total_batch_size = config.batch_size_per_gpu * num_gpus
    
    # Set primary device (first GPU)
    device = torch.device('cuda:0')
    logger.info(f"Primary device: {device}")
    
    # Log GPU information
    for i in range(num_gpus):
        gpu = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {gpu.name}, Memory: {gpu.total_memory / 1e9:.2f} GB")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.vocab_size = tokenizer.vocab_size
    
    # Create datasets
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
    
    # Load test dataset
    test_dataset = AGNewsDataset('test', tokenizer, config, max_samples=7600)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Create dataloaders optimized for DataParallel
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.total_batch_size,  # Total batch size across GPUs
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.total_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        prefetch_factor=config.prefetch_factor,
        persistent_workers=config.persistent_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.total_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )
    
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Create model
    logger.info("\n" + "="*60)
    logger.info("Initializing TAN Model for Classification")
    logger.info("="*60)
    
    # Create TopoformerConfig
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
    
    # Create model with gradient checkpointing
    model = TopoformerForClassification(
        topo_config,
        config.num_labels,
        use_gradient_checkpointing=config.gradient_checkpointing
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = DataParallelTrainer(
        model=model,
        config=config,
        device=device
    )
    
    # Train model
    logger.info("\n" + "="*60)
    logger.info(f"Starting DataParallel Training on {config.num_gpus} GPUs")
    logger.info("="*60)
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.num_epochs,
        save_dir=config.save_dir
    )
    
    # Test best model
    logger.info("\n" + "="*60)
    logger.info("Testing Best Model")
    logger.info("="*60)
    
    # Load best model
    best_model_path = Path(config.save_dir) / 'best_model.pt'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model = TopoformerForClassification(
            topo_config,
            config.num_labels,
            use_gradient_checkpointing=False  # Disable for testing
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Wrap in DataParallel for testing if multiple GPUs
        if num_gpus > 1:
            model = DataParallel(model, device_ids=[0, 1] if num_gpus >= 2 else list(range(num_gpus)))
        
        test_results = test_model(model, test_loader, device, Path(config.save_dir))
    else:
        logger.warning("Best model not found. Skipping testing.")
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Results saved to: {config.save_dir}")
    
    # Final memory report
    log_memory_usage("Final")


if __name__ == "__main__":
    main()
