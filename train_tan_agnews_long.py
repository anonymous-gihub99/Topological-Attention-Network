"""
Train TAN/Topoformer on AG News Dataset - Long Context Classification with DataParallel
Implements training for news classification with artificially extended sequences
Uses document augmentation techniques to create longer contexts

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
        logging.FileHandler('agnews_long_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class AGNewsLongConfig:
    """Configuration for AG News long-context training"""
    # Data parameters
    data_dir: str = './agnews_data'
    save_dir: str = './agnews_models'
    cache_dir: str = './cache'
    
    # Model parameters
    vocab_size: int = 50000
    embed_dim: int = 768
    num_layers: int = 8  # More layers for long context
    num_heads: int = 12
    dropout: float = 0.1
    k_neighbors: int = 48  # More neighbors for long sequences
    use_topology: bool = True
    
    # Long context parameters
    min_seq_len: int = 512
    max_seq_len: int = 2048  # Start with 2K, can extend
    context_extension_methods: List[str] = None
    
    # Training parameters
    batch_size: int = 8  # Smaller batch for long sequences
    gradient_accumulation_steps: int = 8  # More accumulation
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_epochs: int = 10
    max_grad_norm: float = 1.0
    
    # Classification
    num_labels: int = 4  # AG News has 4 classes
    
    # Optimization
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True  # Important for long sequences
    
    # Hardware
    num_workers: int = 4  # Less workers for larger batches
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    def __post_init__(self):
        if self.context_extension_methods is None:
            self.context_extension_methods = ['paraphrase', 'context', 'retrieval', 'duplicate']


class TopoformerForLongContext(nn.Module):
    """TAN/Topoformer adapted for long-context classification"""
    
    def __init__(self, config: TopoformerConfig, num_labels: int):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        
        # Embeddings with positional encoding for long sequences
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.segment_embeddings = nn.Embedding(4, config.embed_dim)  # For document segments
        self.embedding_norm = nn.LayerNorm(config.embed_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        
        # Hierarchical pooling
        self.local_pool_size = 64  # Pool every 64 tokens
        self.local_pooler = nn.Conv1d(config.embed_dim, config.embed_dim, 
                                      kernel_size=self.local_pool_size, 
                                      stride=self.local_pool_size)
        
        # Topoformer layers
        self.layers = nn.ModuleList([
            TopoformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Global pooling
        self.pooler = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.Tanh(),
            nn.Dropout(config.dropout)
        )
        
        # Classifier
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
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with hierarchical processing for long sequences
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            segment_ids: [batch_size, seq_len] - segment indicators
            labels: [batch_size]
            use_cache: Whether to cache intermediate states
            
        Returns:
            Dictionary with loss and/or logits
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position ids
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Default segment ids if not provided
        if segment_ids is None:
            # Create segments (0, 1, 2, 3) for quarters of the document
            segment_ids = torch.zeros_like(input_ids)
            segment_size = seq_len // 4
            for i in range(4):
                start_idx = i * segment_size
                end_idx = (i + 1) * segment_size if i < 3 else seq_len
                segment_ids[:, start_idx:end_idx] = i
        
        # Embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        segment_embeds = self.segment_embeddings(segment_ids)
        
        embeddings = token_embeds + position_embeds + segment_embeds
        embeddings = self.embedding_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        
        # Hierarchical processing for very long sequences
        if seq_len > 1024:
            # Apply local pooling first
            embeddings_transposed = embeddings.transpose(1, 2)  # [batch, embed_dim, seq_len]
            pooled_embeddings = self.local_pooler(embeddings_transposed)
            pooled_embeddings = pooled_embeddings.transpose(1, 2)  # [batch, pooled_seq_len, embed_dim]
            
            # Adjust attention mask
            if attention_mask is not None:
                pooled_mask = F.max_pool1d(
                    attention_mask.unsqueeze(1).float(),
                    kernel_size=self.local_pool_size,
                    stride=self.local_pool_size
                ).squeeze(1).long()
            else:
                pooled_mask = None
            
            hidden_states = pooled_embeddings
            mask = pooled_mask
        else:
            hidden_states = embeddings
            mask = attention_mask
        
        # Apply Topoformer layers with gradient checkpointing
        for i, layer in enumerate(self.layers):
            if self.training and self.config.num_layers > 4 and i > 1:
                # Use gradient checkpointing for middle layers
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, mask
                )
            else:
                hidden_states = layer(hidden_states, mask)
        
        # Global pooling with attention weights
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
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


class AGNewsLongDataset(Dataset):
    """Dataset class for AG News with context extension"""
    
    def __init__(
        self,
        split: str,
        tokenizer,
        config: AGNewsLongConfig,
        augment: bool = True
    ):
        """
        Initialize dataset with context extension
        
        Args:
            split: 'train' or 'test'
            tokenizer: Tokenizer to use
            config: Configuration object
            augment: Whether to augment texts to create longer contexts
        """
        self.split = split
        self.tokenizer = tokenizer
        self.config = config
        self.augment = augment and (split == 'train')
        
        # Load dataset
        logger.info(f"Loading AG News {split} dataset...")
        self.dataset = load_dataset(
            'fancyzhx/ag_news',
            split=split,
            cache_dir=config.cache_dir,
            trust_remote_code=True
        )
        
        # Class labels
        self.label_names = ['World', 'Sports', 'Business', 'Sci/Tech']
        
        logger.info(f"Loaded {len(self.dataset)} samples")
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def _extend_context(self, text: str, label: int) -> str:
        """
        Extend text context using various augmentation techniques
        
        Args:
            text: Original text
            label: Text label for context-aware extension
            
        Returns:
            Extended text
        """
        extended_texts = [text]
        current_length = len(self.tokenizer.tokenize(text))
        target_length = random.randint(self.config.min_seq_len, self.config.max_seq_len)
        
        methods = self.config.context_extension_methods.copy()
        random.shuffle(methods)
        
        for method in methods:
            if current_length >= target_length:
                break
                
            if method == 'paraphrase':
                # Add paraphrased version
                paraphrase = self._paraphrase(text)
                extended_texts.append(paraphrase)
                
            elif method == 'context':
                # Add contextual information based on label
                context = self._get_label_context(label)
                extended_texts.append(context)
                
            elif method == 'retrieval':
                # Retrieve and add similar samples from the same class
                similar_text = self._get_similar_sample(label)
                if similar_text:
                    extended_texts.append(similar_text)
                    
            elif method == 'duplicate':
                # Duplicate with slight modifications
                modified = self._slight_modification(text)
                extended_texts.append(modified)
            
            # Update current length
            current_text = " [SEP] ".join(extended_texts)
            current_length = len(self.tokenizer.tokenize(current_text))
        
        return " [SEP] ".join(extended_texts)
    
    def _paraphrase(self, text: str) -> str:
        """Simple paraphrasing by word replacement"""
        words = text.split()
        num_changes = max(1, len(words) // 10)
        
        for _ in range(num_changes):
            if len(words) > 0:
                idx = random.randint(0, len(words) - 1)
                # Simple synonym replacement (in practice, use a proper paraphraser)
                synonyms = {
                    'good': 'excellent', 'bad': 'poor', 'big': 'large',
                    'small': 'tiny', 'fast': 'quick', 'slow': 'sluggish'
                }
                word = words[idx].lower()
                if word in synonyms:
                    words[idx] = synonyms[word]
        
        return ' '.join(words)
    
    def _get_label_context(self, label: int) -> str:
        """Get contextual information for each label"""
        contexts = {
            0: "This article discusses world events, international relations, global politics, "
               "and matters affecting multiple countries and regions worldwide.",
            1: "This content covers sports events, athletic competitions, team performances, "
               "player statistics, and sporting achievements across various disciplines.",
            2: "This piece examines business developments, economic trends, corporate news, "
               "financial markets, and commercial activities in various industries.",
            3: "This text explores scientific discoveries, technological innovations, "
               "research breakthroughs, and advances in computing and engineering."
        }
        return contexts.get(label, "This is a news article covering recent events and developments.")
    
    def _get_similar_sample(self, label: int) -> Optional[str]:
        """Get a similar sample from the same class"""
        # Find samples with the same label
        same_label_indices = [
            i for i in range(len(self.dataset)) 
            if self.dataset[i]['label'] == label and i != self.current_idx
        ]
        
        if same_label_indices:
            idx = random.choice(same_label_indices)
            return self.dataset[idx]['text']
        return None
    
    def _slight_modification(self, text: str) -> str:
        """Slightly modify text (e.g., change tense, add connectors)"""
        connectors = [
            "Furthermore,", "Additionally,", "Moreover,", "In addition,",
            "As a result,", "Consequently,", "Therefore,", "Thus,"
        ]
        connector = random.choice(connectors)
        return f"{connector} {text}"
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample with extended context"""
        self.current_idx = idx  # For similar sample retrieval
        item = self.dataset[idx]
        
        # Get text and label
        text = item['text']
        label = item['label']
        
        # Extend context if augmenting
        if self.augment:
            text = self._extend_context(text, label)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_seq_len,
            return_tensors='pt'
        )
        
        # Create segment ids based on [SEP] tokens
        input_ids = encoding['input_ids'].squeeze(0)
        sep_token_id = self.tokenizer.sep_token_id
        segment_ids = torch.zeros_like(input_ids)
        
        if sep_token_id:
            sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
            for i, pos in enumerate(sep_positions[:3]):  # Max 4 segments
                if i < len(sep_positions) - 1:
                    segment_ids[pos:sep_positions[i+1]] = i + 1
                else:
                    segment_ids[pos:] = i + 1
        
        return {
            'input_ids': input_ids,
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'segment_ids': segment_ids,
            'labels': torch.tensor(label, dtype=torch.long),
            'original_text': item['text']  # Keep original for analysis
        }


class LongContextTrainer:
    """Trainer for long-context classification with DataParallel support"""
    
    def __init__(
        self,
        model: nn.Module,
        config: AGNewsLongConfig,
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
        
        # Optimizer with different learning rates for different components
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
        """Train for one epoch with gradient accumulation"""
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
            segment_ids = batch['segment_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Mixed precision training
            if self.config.use_mixed_precision:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        segment_ids=segment_ids,
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
                    segment_ids=segment_ids,
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
                segment_ids = batch['segment_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    segment_ids=segment_ids,
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
        
        plt.suptitle('AG News Long-Context Training Curves', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


def test_model_at_different_lengths(
    model: nn.Module,
    test_dataset: AGNewsLongDataset,
    config: AGNewsLongConfig,
    device: torch.device,
    save_dir: Path,
    test_lengths: List[int] = None
) -> Dict[str, Any]:
    """Test model performance at different sequence lengths"""
    
    if test_lengths is None:
        test_lengths = [512, 1024, 2048, 4096]
    
    results_by_length = {}
    
    for max_len in test_lengths:
        logger.info(f"\nTesting at max length: {max_len}")
        
        # Update config for this length
        test_config = AGNewsLongConfig()
        test_config.max_seq_len = max_len
        test_config.min_seq_len = min(256, max_len // 2)
        
        # Create dataset with specific length
        length_dataset = AGNewsLongDataset(
            split='test',
            tokenizer=test_dataset.tokenizer,
            config=test_config,
            augment=True  # Augment test data to reach target length
        )
        
        # Create dataloader
        test_loader = DataLoader(
            length_dataset,
            batch_size=max(1, 8192 // max_len),  # Adjust batch size based on length
            shuffle=False,
            num_workers=2
        )
        
        # Test model
        model.eval()
        all_preds = []
        all_labels = []
        total_time = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Testing at {max_len} tokens"):
                start_time = time.time()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                segment_ids = batch['segment_ids'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    segment_ids=segment_ids
                )
                
                total_time += time.time() - start_time
                
                preds = torch.argmax(outputs['logits'], dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        # Store results
        results_by_length[max_len] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_macro': f1_macro,
            'time_per_sample': total_time / len(all_labels),
            'num_samples': len(all_labels)
        }
        
        logger.info(f"Length {max_len} - Accuracy: {accuracy:.4f}, F1: {f1_macro:.4f}")
    
    # Create performance vs length plot
    lengths = list(results_by_length.keys())
    accuracies = [results_by_length[l]['accuracy'] for l in lengths]
    f1_scores = [results_by_length[l]['f1_macro'] for l in lengths]
    times = [results_by_length[l]['time_per_sample'] for l in lengths]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Accuracy vs Length
    ax = axes[0]
    ax.plot(lengths, accuracies, 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Sequence Length')
    ax.grid(True, alpha=0.3)
    
    # F1 vs Length
    ax = axes[1]
    ax.plot(lengths, f1_scores, 'g-s', linewidth=2, markersize=8)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('F1-Macro')
    ax.set_title('F1-Macro vs Sequence Length')
    ax.grid(True, alpha=0.3)
    
    # Time vs Length
    ax = axes[2]
    ax.plot(lengths, times, 'r-^', linewidth=2, markersize=8)
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Time per Sample (s)')
    ax.set_title('Inference Time vs Sequence Length')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Performance Analysis at Different Sequence Lengths', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / 'length_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results_by_length


def main():
    """Main training script"""
    # Configuration
    config = AGNewsLongConfig()
    
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
    logger.info("Loading AG News Dataset")
    logger.info("="*60)
    
    # Split training data for validation
    full_train_dataset = AGNewsLongDataset('train', tokenizer, config, augment=True)
    
    # Create train/val split (90/10)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size]
    )
    
    test_dataset = AGNewsLongDataset('test', tokenizer, config, augment=False)
    
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
    logger.info("Initializing TAN Model for Long Context")
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
    
    model = TopoformerForLongContext(topo_config, config.num_labels)
    
    # Enable gradient checkpointing if specified
    if config.gradient_checkpointing:
        model.config.num_layers = config.num_layers  # Ensure config is set
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = LongContextTrainer(
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
    
    # Test model at different lengths
    logger.info("\n" + "="*60)
    logger.info("Testing at Different Sequence Lengths")
    logger.info("="*60)
    
    # Load best model
    best_model_path = Path(config.save_dir) / 'best_model.pt'
    checkpoint = torch.load(best_model_path, map_location=device)
    
    model = TopoformerForLongContext(topo_config, config.num_labels)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Test at different lengths
    length_results = test_model_at_different_lengths(
        model=model,
        test_dataset=test_dataset,
        config=config,
        device=device,
        save_dir=Path(config.save_dir),
        test_lengths=[512, 1024, 2048]  # Test at these lengths
    )
    
    # Save results
    with open(Path(config.save_dir) / 'length_test_results.json', 'w') as f:
        json.dump(length_results, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("Training and Testing Complete!")
    logger.info("="*60)
    logger.info(f"Results saved to: {config.save_dir}")


if __name__ == "__main__":
    main()