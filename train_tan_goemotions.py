"""
Train TAN/Topoformer on GoEmotions Dataset - Multi-label Classification with DataParallel
Implements training, validation, and testing for emotion classification
Uses the simplified subset with 27 emotion labels

Author: TAN Research Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import autocast, GradScaler

import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from tqdm import tqdm
import time
from datetime import datetime
import gc

# Metrics
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    hamming_loss, jaccard_score, multilabel_confusion_matrix,
    classification_report
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
    raise ImportError("Topoformer modules not found. Please ensure streamlined_topoformer.py is available.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('goemotions_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class GoEmotionsConfig:
    """Configuration for GoEmotions training"""
    # Data parameters
    data_dir: str = './goemotions_data'
    save_dir: str = './goemotions_models'
    cache_dir: str = './cache'
    
    # Model parameters
    vocab_size: int = 50268
    embed_dim: int = 768
    num_layers: int = 8
    num_heads: int = 14
    max_seq_len: int = 128  # GoEmotions has short texts
    dropout: float = 0.1
    k_neighbors: int = 64
    use_topology: bool = True
    
    # Training parameters
    batch_size: int = 64  # Larger batch for short texts
    gradient_accumulation_steps: int = 2
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_epochs: int = 15
    max_grad_norm: float = 1.0
    
    # Multi-label specific
    num_labels: int = 27  # Simplified subset
    threshold: float = 0.5  # Prediction threshold
    use_class_weights: bool = True
    
    # Optimization
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = False
    
    # Hardware
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 2


class TopoformerForMultiLabelClassification(nn.Module):
    """TAN/Topoformer for multi-label emotion classification"""
    
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
        
        # Pooling
        self.pooler = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.Tanh()
        )
        
        # Multi-label classifier
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
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
        labels: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, num_labels] - multi-hot encoded
            return_hidden_states: Whether to return all hidden states
            
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
        all_hidden_states = [] if return_hidden_states else None
        
        for layer in self.layers:
            if return_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states = layer(hidden_states, attention_mask)
        
        # Pooling (mean pooling with attention mask)
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
            loss_fct = nn.BCEWithLogitsLoss(reduction='mean')
            loss = loss_fct(logits, labels.float())
            outputs['loss'] = loss
        
        if return_hidden_states:
            outputs['hidden_states'] = all_hidden_states
            
        return outputs


class GoEmotionsDataset(Dataset):
    """Dataset class for GoEmotions"""
    
    def __init__(
        self,
        split: str,
        tokenizer,
        max_length: int = 128,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize dataset
        
        Args:
            split: 'train', 'validation', or 'test'
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            cache_dir: Cache directory for dataset
        """
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load dataset
        logger.info(f"Loading GoEmotions {split} dataset...")
        self.dataset = load_dataset(
            'google-research-datasets/go_emotions',
            'simplified',
            split='train',
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        # Get label information
        self.label_names = self._get_label_names()
        self.num_labels = len(self.label_names)
        
        logger.info(f"Loaded {len(self.dataset)} samples with {self.num_labels} labels")
        
    def _get_label_names(self) -> List[str]:
        """Get emotion label names"""
        # GoEmotions simplified has 27 emotions + neutral
        emotions = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval',
            'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
            'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
            'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
            'pride', 'realization', 'relief', 'remorse', 'sadness',
            'surprise', 'neutral'
        ]
        return emotions[:27]  # Use first 27 for simplified
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        item = self.dataset[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Process labels (multi-hot encoding)
        labels = torch.zeros(self.num_labels)
        for label_idx in item['labels']:
            if label_idx < self.num_labels:  # Safety check
                labels[label_idx] = 1
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels,
            'text': item['text']  # Keep for analysis
        }


class MultiLabelTrainer:
    """Trainer for multi-label classification with DataParallel support"""
    
    def __init__(
        self,
        model: nn.Module,
        config: GoEmotionsConfig,
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
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Metrics tracking
        self.train_history = []
        self.val_history = []
        self.best_val_f1 = 0.0
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        scheduler=None
    ) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} - Training")
        
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
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    if scheduler is not None:
                        scheduler.step()
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
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if scheduler is not None:
                        scheduler.step()
            
            # Track metrics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # Get predictions
            with torch.no_grad():
                preds = torch.sigmoid(outputs['logits']) > self.config.threshold
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item() * self.config.gradient_accumulation_steps:.4f}'
            })
        
        # Calculate metrics
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        metrics = self._calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(train_loader)
        
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
                
                # Get predictions
                preds = torch.sigmoid(outputs['logits']) > self.config.threshold
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
        # Calculate metrics
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        metrics = self._calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(val_loader)
        
        return metrics
    
    def _calculate_metrics(
        self,
        labels: np.ndarray,
        preds: np.ndarray
    ) -> Dict[str, float]:
        """Calculate multi-label metrics"""
        # Macro metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )
        
        # Micro metrics
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            labels, preds, average='micro', zero_division=0
        )
        
        # Hamming loss
        hamming = hamming_loss(labels, preds)
        
        # Subset accuracy (exact match)
        subset_acc = accuracy_score(labels, preds)
        
        # Jaccard score (IoU)
        jaccard = jaccard_score(labels, preds, average='samples', zero_division=0)
        
        return {
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'hamming_loss': hamming,
            'subset_accuracy': subset_acc,
            'jaccard_score': jaccard
        }
    
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
                       f"F1-Macro: {train_metrics['f1_macro']:.4f}, "
                       f"F1-Micro: {train_metrics['f1_micro']:.4f}")
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"F1-Macro: {val_metrics['f1_macro']:.4f}, "
                       f"F1-Micro: {val_metrics['f1_micro']:.4f}, "
                       f"Hamming Loss: {val_metrics['hamming_loss']:.4f}")
            
            # Save best model
            if val_metrics['f1_macro'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1_macro']
                self.save_model(save_path / 'best_model.pt', epoch, val_metrics)
                logger.info(f"Saved best model with F1-Macro: {self.best_val_f1:.4f}")
            
            # Save checkpoint
            if epoch % 5 == 0:
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
        
        logger.info(f"\nTraining complete! Best validation F1-Macro: {self.best_val_f1:.4f}")
    
    def save_model(self, path: Path, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        model_to_save = self.model.module if isinstance(self.model, DataParallel) else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__,
            'best_val_f1': self.best_val_f1
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def plot_training_curves(self, save_dir: Path):
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
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
        
        # F1-Macro
        ax = axes[0, 1]
        ax.plot(epochs, [h['f1_macro'] for h in self.train_history], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, [h['f1_macro'] for h in self.val_history], 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1-Macro')
        ax.set_title('F1-Macro Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # F1-Micro
        ax = axes[0, 2]
        ax.plot(epochs, [h['f1_micro'] for h in self.train_history], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, [h['f1_micro'] for h in self.val_history], 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1-Micro')
        ax.set_title('F1-Micro Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Hamming Loss
        ax = axes[1, 0]
        ax.plot(epochs, [h['hamming_loss'] for h in self.val_history], 'g-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Hamming Loss')
        ax.set_title('Validation Hamming Loss')
        ax.grid(True, alpha=0.3)
        
        # Subset Accuracy
        ax = axes[1, 1]
        ax.plot(epochs, [h['subset_accuracy'] for h in self.val_history], 'm-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Subset Accuracy')
        ax.set_title('Validation Subset Accuracy')
        ax.grid(True, alpha=0.3)
        
        # Jaccard Score
        ax = axes[1, 2]
        ax.plot(epochs, [h['jaccard_score'] for h in self.val_history], 'c-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Jaccard Score')
        ax.set_title('Validation Jaccard Score')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('GoEmotions Multi-Label Training Curves', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    config: GoEmotionsConfig,
    device: torch.device,
    save_dir: Path
) -> Dict[str, float]:
    """Test the trained model and save detailed results"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            probs = torch.sigmoid(outputs['logits'])
            preds = probs > config.threshold
            
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate results
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Calculate comprehensive metrics
    trainer = MultiLabelTrainer(model, config, device, use_data_parallel=False)
    metrics = trainer._calculate_metrics(all_labels, all_preds)
    
    # Per-class metrics
    per_class_metrics = []
    dataset = test_loader.dataset
    for i, label_name in enumerate(dataset.label_names):
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels[:, i], all_preds[:, i], average='binary', zero_division=0
        )
        per_class_metrics.append({
            'label': label_name,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support[1]  # Positive class support
        })
    
    # Save results
    results = {
        'overall_metrics': metrics,
        'per_class_metrics': per_class_metrics,
        'predictions': all_preds.tolist(),
        'probabilities': all_probs.tolist(),
        'labels': all_labels.tolist()
    }
    
    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create per-class performance plot
    df_per_class = pd.DataFrame(per_class_metrics)
    df_per_class = df_per_class.sort_values('f1', ascending=False)
    
    plt.figure(figsize=(15, 8))
    x = range(len(df_per_class))
    plt.bar(x, df_per_class['f1'].values, color='skyblue', edgecolor='black')
    plt.xticks(x, df_per_class['label'].values, rotation=45, ha='right')
    plt.xlabel('Emotion Label')
    plt.ylabel('F1 Score')
    plt.title('Per-Class F1 Scores on GoEmotions Test Set')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_dir / 'per_class_f1.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("\nTest Results:")
    logger.info(f"F1-Macro: {metrics['f1_macro']:.4f}")
    logger.info(f"F1-Micro: {metrics['f1_micro']:.4f}")
    logger.info(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
    logger.info(f"Subset Accuracy: {metrics['subset_accuracy']:.4f}")
    logger.info(f"Jaccard Score: {metrics['jaccard_score']:.4f}")
    
    return metrics


def main():
    """Main training script"""
    # Configuration
    config = GoEmotionsConfig()
    
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
    config.vocab_size = tokenizer.vocab_size
    
    # Create datasets
    logger.info("\n" + "="*60)
    logger.info("Loading GoEmotions Dataset")
    logger.info("="*60)
    
    train_dataset = GoEmotionsDataset('train', tokenizer, config.max_seq_len, config.cache_dir)
    val_dataset = GoEmotionsDataset('validation', tokenizer, config.max_seq_len, config.cache_dir)
    test_dataset = GoEmotionsDataset('test', tokenizer, config.max_seq_len, config.cache_dir)
    
    # Update config with actual number of labels
    config.num_labels = train_dataset.num_labels
    logger.info(f"Number of emotion labels: {config.num_labels}")
    
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
        batch_size=config.batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        prefetch_factor=config.prefetch_factor
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # Create model
    logger.info("\n" + "="*60)
    logger.info("Initializing TAN Model")
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
    
    model = TopoformerForMultiLabelClassification(topo_config, config.num_labels)
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = MultiLabelTrainer(
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
    checkpoint = torch.load(best_model_path, map_location=device)
    
    model = TopoformerForMultiLabelClassification(topo_config, config.num_labels)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Test
    test_metrics = test_model(
        model=model,
        test_loader=test_loader,
        config=config,
        device=device,
        save_dir=Path(config.save_dir)
    )
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Final test F1-Macro: {test_metrics['f1_macro']:.4f}")
    logger.info(f"Final test F1-Micro: {test_metrics['f1_micro']:.4f}")
    logger.info(f"Results saved to: {config.save_dir}")


if __name__ == "__main__":
    main()
