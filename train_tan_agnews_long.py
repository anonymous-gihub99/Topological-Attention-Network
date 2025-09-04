"""
Train TAN/Topoformer on AG News Dataset - Original Dataset Classification with DataParallel
Implements training for news classification using original AG News samples without artificial extension

Author: TAN Research Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, Dataset
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agnews_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AGNewsConfig:
    """Configuration for AG News training"""
    # Data parameters
    data_dir: str = './agnews_data'
    save_dir: str = './agnews_models'
    cache_dir: str = './cache'
    max_samples: int = 90000  # Limit to 90k samples for consistency
    
    # Model parameters
    vocab_size: int = 50268
    embed_dim: int = 768
    num_layers: int = 8
    num_heads: int = 12
    dropout: float = 0.1
    k_neighbors: int = 32  # Reduced for shorter sequences
    use_topology: bool = True
    
    # Sequence parameters - adjusted for original AG News
    max_seq_len: int = 512  # Standard length for AG News
    
    # Training parameters
    batch_size: int = 16  # Larger batch for shorter sequences
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_epochs: int = 7
    max_grad_norm: float = 1.0
    
    # Classification
    num_labels: int = 4  # AG News has 4 classes
    
    # Optimization
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = False  # Not needed for shorter sequences
    
    # Hardware
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2


class TopoformerForClassification(nn.Module):
    """TAN/Topoformer adapted for text classification"""
    
    def __init__(self, config: TopoformerConfig, num_labels: int):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        
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
        """
        Forward pass for classification
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size]
            
        Returns:
            Dictionary with loss and/or logits
        """
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
        
        # Apply Topoformer layers
        hidden_states = embeddings
        for layer in self.layers:
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
        """
        Initialize dataset with original AG News data
        
        Args:
            split: 'train' or 'test'
            tokenizer: Tokenizer to use
            config: Configuration object
            max_samples: Maximum number of samples to use
        """
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
            # Use a deterministic subset for reproducibility
            indices = list(range(0, len(self.dataset), len(self.dataset) // max_samples))[:max_samples]
            self.dataset = self.dataset.select(indices)
            logger.info(f"Limited to {len(self.dataset)} samples")
        
        # Class labels mapping
        self.label_names = ['World', 'Sports', 'Business', 'Sci/Tech']
        
        logger.info(f"Loaded {len(self.dataset)} {split} samples")
        
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
            'text': text  # Keep original text for analysis
        }


class Trainer:
    """Trainer for text classification with DataParallel support"""
    
    def __init__(
        self,
        model: nn.Module,
        config: AGNewsConfig,
        device: torch.device,
        use_data_parallel: bool = True,
        gpu_ids: Optional[List[int]] = None
    ):
        self.config = config
        self.device = device
        
        # Setup DataParallel
        if use_data_parallel and torch.cuda.device_count() > 1:
            if gpu_ids is None:
                gpu_ids = list(range(torch.cuda.device_count()))
            logger.info(f"Using DataParallel on GPUs: {gpu_ids}")
            model = model.to(device)
            self.model = DataParallel(model, device_ids=gpu_ids)
        else:
            self.model = model.to(device)
            logger.info(f"Using single device: {device}")
        
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
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} - Training")
        accumulation_counter = 0
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Mixed precision training
            if self.config.use_mixed_precision:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs['loss']
                    
                    # Handle DataParallel
                    if isinstance(self.model, DataParallel):
                        loss = loss.mean()
                    
                    # Gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs['loss']
                
                # Handle DataParallel
                if isinstance(self.model, DataParallel):
                    loss = loss.mean()
                
                # Gradient accumulation
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
                preds = torch.argmax(outputs['logits'], dim=-1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item() * self.config.gradient_accumulation_steps:.4f}',
                'acc': f'{total_correct / total_samples:.4f}'
            })
        
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
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} - Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
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
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: str
    ):
        """Full training loop"""
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
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"{'='*50}")
            
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
            
            # Save checkpoint
            if epoch % 3 == 0:
                self.save_model(save_path / f'checkpoint_epoch_{epoch}.pt', epoch, val_metrics)
        
        # Save final model
        self.save_model(save_path / 'final_model.pt', num_epochs, val_metrics)
        
        # Plot training curves
        self.plot_training_curves(save_path)
        
        # Save training history
        history = {
            'train': self.train_history,
            'val': self.val_history,
            'config': self.config.__dict__
        }
        with open(save_path / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"\nTraining complete! Best validation accuracy: {self.best_val_acc:.4f}")
    
    def save_model(self, path: Path, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        model_to_save = self.model.module if isinstance(self.model, DataParallel) else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__,
            'best_val_acc': self.best_val_acc
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def plot_training_curves(self, save_dir: Path):
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
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
        ax.set_title('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # F1 Scores
        ax = axes[1, 0]
        ax.plot(epochs, [h['f1_macro'] for h in self.val_history], 'g-', label='F1-Macro', linewidth=2)
        ax.plot(epochs, [h['f1_micro'] for h in self.val_history], 'm-', label='F1-Micro', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Scores (Validation)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Precision/Recall
        ax = axes[1, 1]
        ax.plot(epochs, [h['precision'] for h in self.val_history], 'c-', label='Precision', linewidth=2)
        ax.plot(epochs, [h['recall'] for h in self.val_history], 'y-', label='Recall', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Precision and Recall (Validation)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('AG News Training Curves', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


def test_model(
    model: nn.Module,
    test_dataset: AGNewsDataset,
    device: torch.device,
    save_dir: Path,
    batch_size: int = 32
) -> Dict[str, Any]:
    """Test model on test dataset"""
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    model.eval()
    all_preds = []
    all_labels = []
    
    logger.info("Running final test evaluation...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = torch.argmax(outputs['logits'], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    class_report = classification_report(
        all_labels, all_preds,
        target_names=test_dataset.label_names,
        output_dict=True
    )
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=test_dataset.label_names,
        yticklabels=test_dataset.label_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Results
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report
    }
    
    logger.info(f"Test Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1-Macro: {f1_macro:.4f}")
    logger.info(f"F1-Micro: {f1_micro:.4f}")
    
    return results


def main():
    """Main training script"""
    # Configuration
    config = AGNewsConfig()
    
    # Setup device and GPUs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    logger.info(f"Using device: {device}")
    logger.info(f"Number of GPUs available: {num_gpus}")
    
    # GPU configuration
    gpu_ids = list(range(min(4, num_gpus)))  # Use up to 4 GPUs
    use_data_parallel = num_gpus > 1
    
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
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Load test dataset (original full test set)
    test_dataset = AGNewsDataset('test', tokenizer, config)
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # Create model
    logger.info("\n" + "="*60)
    logger.info("Initializing TAN Model for Classification")
    logger.info("="*60)
    
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
    
    model = TopoformerForClassification(topo_config, config.num_labels)
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        use_data_parallel=use_data_parallel,
        gpu_ids=gpu_ids
    )
    
    # Train model
    logger.info("\n" + "="*60)
    logger.info("Starting Training")
    logger.info("="*60)
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.num_epochs,
        save_dir=config.save_dir
    )
    
    # Test model
    logger.info("\n" + "="*60)
    logger.info("Testing Best Model")
    logger.info("="*60)
    
    # Load best model
    best_model_path = Path(config.save_dir) / 'best_model.pt'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        
        # Create fresh model for testing
        test_model = TopoformerForClassification(topo_config, config.num_labels)
        test_model.load_state_dict(checkpoint['model_state_dict'])
        test_model = test_model.to(device)
        
        # Run test evaluation
        test_results = test_model(
            model=test_model,
            test_dataset=test_dataset,
            device=device,
            save_dir=Path(config.save_dir),
            batch_size=config.batch_size * 2
        )
        
        # Save test results
        with open(Path(config.save_dir) / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        
        logger.info(f"Best validation accuracy: {checkpoint['metrics']['accuracy']:.4f}")
        logger.info(f"Test accuracy: {test_results['accuracy']:.4f}")
    else:
        logger.warning("Best model not found, skipping test evaluation")
    
    logger.info("\n" + "="*60)
    logger.info("Training and Testing Complete!")
    logger.info("="*60)
    logger.info(f"Results saved to: {config.save_dir}")


if __name__ == "__main__":
    main()
