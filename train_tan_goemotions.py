"""
Train TAN/Topoformer on GoEmotions Dataset - Fixed Multi-label Classification with DataParallel
Implements training with proper handling of class imbalance and multi-label issues
Uses Focal Loss and dynamic threshold optimization

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
        logging.FileHandler('goemotions_training_fixed.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class GoEmotionsConfig:
    """Configuration for GoEmotions training - FIXED for class imbalance"""
    # Data parameters
    data_dir: str = './goemotions_data'
    save_dir: str = './goemotions_models_fixed'
    cache_dir: str = './cache'
    
    # Model parameters
    vocab_size: int = 50000
    embed_dim: int = 768
    num_layers: int = 6
    num_heads: int = 8
    max_seq_len: int = 128  # GoEmotions has short texts
    dropout: float = 0.1
    k_neighbors: int = 64
    use_topology: bool = True
    
    # Training parameters - UPDATED
    batch_size: int = 64
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-5  # Lower learning rate
    weight_decay: float = 0.01
    warmup_ratio: float = 0.2  # More warmup
    num_epochs: int = 15  # More epochs for better convergence
    max_grad_norm: float = 1.0
    
    # Multi-label specific - UPDATED
    num_labels: int = 2
    threshold: float = 0.47  # Lower threshold for imbalanced data
    use_class_weights: bool = True
    use_focal_loss: bool = True  # Better for imbalanced multi-label
    use_asymmetric_loss: bool = False  # Alternative to focal loss
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    asymmetric_gamma_neg: float = 4
    asymmetric_gamma_pos: float = 1
    asymmetric_clip: float = 0.05
    label_smoothing: float = 0.1  # Prevent overconfidence
    
    # Dynamic threshold
    use_dynamic_threshold: bool = True
    threshold_search_range: Tuple[float, float] = (0.1, 0.5)
    threshold_search_step: float = 0.05
    
    # Optimization
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = False
    
    # Hardware
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Monitoring
    log_prediction_stats: bool = True
    log_interval: int = 50  # Log every N batches


class FocalLoss(nn.Module):
    """Focal Loss for multi-label classification"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-BCE_loss)  # Probability of being classified correctly
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification - handles class imbalance better"""
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, reduction='mean'):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Calculating Probabilities
        inputs_sigmoid = torch.sigmoid(inputs)
        inputs_sigmoid = torch.clamp(inputs_sigmoid, self.eps, 1 - self.eps)
        
        # Asymmetric Clipping
        inputs_sigmoid_neg = 1 - inputs_sigmoid
        
        if self.clip is not None and self.clip > 0:
            inputs_sigmoid_neg = (inputs_sigmoid_neg + self.clip).clamp(max=1)
        
        # Basic CE calculation
        loss_pos = targets * torch.log(inputs_sigmoid)
        loss_neg = (1 - targets) * torch.log(inputs_sigmoid_neg)
        
        # Asymmetric Focusing
        if self.gamma_pos > 0 and self.gamma_neg > 0:
            loss_pos = loss_pos * (1 - inputs_sigmoid) ** self.gamma_pos
            loss_neg = loss_neg * inputs_sigmoid ** self.gamma_neg
        
        # Final loss
        loss = -loss_pos - loss_neg
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class TopoformerForMultiLabelClassification(nn.Module):
    """TAN/Topoformer for multi-label emotion classification - FIXED"""
    
    def __init__(self, config: TopoformerConfig, num_labels: int, class_config: GoEmotionsConfig):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        self.class_config = class_config
        
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
        
        # Multi-label classifier with label smoothing initialization
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim // 2, num_labels)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Initialize loss functions
        if class_config.use_focal_loss:
            self.loss_fn = FocalLoss(
                alpha=class_config.focal_alpha,
                gamma=class_config.focal_gamma
            )
        elif class_config.use_asymmetric_loss:
            self.loss_fn = AsymmetricLoss(
                gamma_neg=class_config.asymmetric_gamma_neg,
                gamma_pos=class_config.asymmetric_gamma_pos,
                clip=class_config.asymmetric_clip
            )
        else:
            self.loss_fn = None  # Will use weighted BCE
        
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                # Bias initialization for multi-label (slightly negative to encourage some predictions)
                if module.out_features == self.num_labels:
                    module.bias.data.fill_(-2.0)  # Start with negative bias
                else:
                    module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pos_weights: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with improved loss computation
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
            # Apply label smoothing if configured
            if self.class_config.label_smoothing > 0:
                smooth_labels = labels * (1 - self.class_config.label_smoothing) + \
                               self.class_config.label_smoothing / 2
            else:
                smooth_labels = labels
            
            if self.loss_fn is not None:
                loss = self.loss_fn(logits, smooth_labels.float())
            else:
                # Use weighted BCE
                if pos_weights is not None:
                    pos_weights = pos_weights.to(device)
                    loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction='mean')
                else:
                    loss_fct = nn.BCEWithLogitsLoss(reduction='mean')
                loss = loss_fct(logits, smooth_labels.float())
            
            outputs['loss'] = loss
        
        if return_hidden_states:
            outputs['hidden_states'] = all_hidden_states
            
        return outputs


class GoEmotionsDataset(Dataset):
    """Dataset class for GoEmotions with class weight calculation"""
    
    def __init__(
        self,
        split: str,
        tokenizer,
        max_length: int = 128,
        cache_dir: Optional[str] = None
    ):
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
        
        # Calculate class statistics
        self.class_weights = self._calculate_class_weights()
        
        logger.info(f"Loaded {len(self.dataset)} samples with {self.num_labels} labels")
        logger.info(f"Label distribution: min count = {self.class_counts.min()}, max count = {self.class_counts.max()}")
        
    def _get_label_names(self) -> List[str]:
        """Get emotion label names"""
        emotions = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval',
            'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
            'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear',
            'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
            'pride', 'realization', 'relief', 'remorse', 'sadness',
            'surprise', 'neutral'
        ]
        return emotions[:27]  # Use first 27 for simplified
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalance"""
        class_counts = np.zeros(self.num_labels)
        
        for item in self.dataset:
            for label_idx in item['labels']:
                if label_idx < self.num_labels:
                    class_counts[label_idx] += 1
        
        # Calculate weights (inverse frequency)
        total_samples = len(self.dataset)
        neg_counts = total_samples - class_counts
        
        # Avoid division by zero and calculate balanced weights
        pos_weights = neg_counts / (class_counts + 1)
        
        # Normalize weights to prevent extreme values
        pos_weights = pos_weights / pos_weights.mean()
        pos_weights = np.clip(pos_weights, 0.5, 10.0)  # Clip to reasonable range
        
        self.class_counts = class_counts
        
        logger.info(f"Class weights range: [{pos_weights.min():.2f}, {pos_weights.max():.2f}]")
        
        return torch.tensor(pos_weights, dtype=torch.float32)
    
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


def check_predictions_distribution(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.3) -> Dict:
    """Monitor prediction statistics to detect issues"""
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        
        # Calculate statistics
        total_predictions = preds.sum().item()
        total_possible = preds.numel()
        avg_preds_per_sample = preds.sum(dim=1).mean().item()
        max_prob = probs.max().item()
        min_prob = probs.min().item()
        mean_prob = probs.mean().item()
        
        # Check if model is predicting any positives
        samples_with_predictions = (preds.sum(dim=1) > 0).sum().item()
        samples_total = preds.size(0)
        
        # True positive rate
        true_positives = (preds * labels).sum().item()
        total_true_labels = labels.sum().item()
        
        return {
            'total_positive_predictions': total_predictions,
            'prediction_rate': total_predictions / total_possible,
            'avg_predictions_per_sample': avg_preds_per_sample,
            'samples_with_predictions': samples_with_predictions,
            'samples_with_predictions_pct': samples_with_predictions / samples_total,
            'max_probability': max_prob,
            'min_probability': min_prob,
            'mean_probability': mean_prob,
            'true_positive_rate': true_positives / (total_true_labels + 1e-8)
        }


def find_optimal_threshold(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    search_range: Tuple[float, float] = (0.1, 0.5),
    search_step: float = 0.05
) -> Tuple[float, float]:
    """Find optimal threshold using validation set"""
    model.eval()
    all_probs = []
    all_labels = []
    
    logger.info("Finding optimal threshold...")
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting predictions"):
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(inputs, masks)
            probs = torch.sigmoid(outputs['logits'])
            
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
    
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    
    # Try different thresholds
    best_threshold = search_range[0]
    best_f1 = 0
    threshold_results = []
    
    for threshold in np.arange(search_range[0], search_range[1] + search_step, search_step):
        preds = (all_probs > threshold).float()
        
        # Calculate metrics
        f1_macro = f1_score(all_labels.numpy(), preds.numpy(), average='macro', zero_division=0)
        f1_micro = f1_score(all_labels.numpy(), preds.numpy(), average='micro', zero_division=0)
        hamming = hamming_loss(all_labels.numpy(), preds.numpy())
        
        # Count predictions
        total_preds = preds.sum().item()
        avg_preds_per_sample = preds.sum(dim=1).mean().item()
        
        threshold_results.append({
            'threshold': threshold,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'hamming_loss': hamming,
            'total_predictions': total_preds,
            'avg_predictions_per_sample': avg_preds_per_sample
        })
        
        if f1_macro > best_f1:
            best_f1 = f1_macro
            best_threshold = threshold
    
    # Log results
    logger.info(f"Optimal threshold: {best_threshold:.3f} with F1-Macro: {best_f1:.4f}")
    for res in threshold_results:
        logger.info(f"  Threshold {res['threshold']:.2f}: F1-Macro={res['f1_macro']:.3f}, "
                   f"F1-Micro={res['f1_micro']:.3f}, Avg Preds={res['avg_predictions_per_sample']:.2f}")
    
    return best_threshold, best_f1


class MultiLabelTrainer:
    """Trainer for multi-label classification with improved monitoring"""
    
    def __init__(
        self,
        model: nn.Module,
        config: GoEmotionsConfig,
        device: torch.device,
        class_weights: torch.Tensor = None,
        use_data_parallel: bool = True,
        gpu_ids: Optional[List[int]] = None
    ):
        self.config = config
        self.device = device
        self.class_weights = class_weights
        
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
        
        # Optimizer with lower learning rate
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            eps=1e-8
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Metrics tracking
        self.train_history = []
        self.val_history = []
        self.best_val_f1 = 0.0
        self.current_threshold = config.threshold
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        scheduler=None
    ) -> Dict[str, float]:
        """Train for one epoch with monitoring"""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        batch_prediction_stats = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} - Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Pass class weights if available
            pos_weights = self.class_weights.to(self.device) if self.class_weights is not None else None
            
            # Mixed precision training
            if self.config.use_mixed_precision:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        pos_weights=pos_weights
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
                    labels=labels,
                    pos_weights=pos_weights
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
            
            # Get predictions and monitor
            with torch.no_grad():
                logits = outputs['logits']
                
                # Monitor prediction distribution
                if self.config.log_prediction_stats and batch_idx % self.config.log_interval == 0:
                    pred_stats = check_predictions_distribution(logits, labels, self.current_threshold)
                    batch_prediction_stats.append(pred_stats)
                    
                    # Log if model is not predicting anything
                    if pred_stats['samples_with_predictions_pct'] < 0.1:
                        logger.warning(f"Low prediction rate: {pred_stats['samples_with_predictions_pct']:.2%} "
                                     f"of samples have predictions")
                
                preds = torch.sigmoid(logits) > self.current_threshold
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item() * self.config.gradient_accumulation_steps:.4f}',
                'pred_rate': f"{pred_stats['prediction_rate']:.2%}" if batch_prediction_stats else "N/A"
            })
        
        # Calculate metrics
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        metrics = self._calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(train_loader)
        
        # Add prediction statistics
        if batch_prediction_stats:
            avg_stats = {
                key: np.mean([s[key] for s in batch_prediction_stats])
                for key in batch_prediction_stats[0].keys()
            }
            metrics.update({f'pred_{k}': v for k, v in avg_stats.items()})
        
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
                
                pos_weights = self.class_weights.to(self.device) if self.class_weights is not None else None
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    pos_weights=pos_weights
                )
                
                loss = outputs['loss']
                if isinstance(self.model, DataParallel):
                    loss = loss.mean()
                
                total_loss += loss.item()
                
                # Get predictions with current threshold
                preds = torch.sigmoid(outputs['logits']) > self.current_threshold
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
        
        # Calculate metrics
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        
        metrics = self._calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(val_loader)
        metrics['threshold'] = self.current_threshold
        
        return metrics
    
    def _calculate_metrics(
        self,
        labels: np.ndarray,
        preds: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive multi-label metrics"""
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
        
        # Coverage metrics
        total_predictions = preds.sum()
        total_true_labels = labels.sum()
        avg_predictions_per_sample = preds.sum(axis=1).mean()
        samples_with_predictions = (preds.sum(axis=1) > 0).mean()
        
        return {
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'hamming_loss': hamming,
            'subset_accuracy': subset_acc,
            'jaccard_score': jaccard,
            'total_predictions': total_predictions,
            'avg_predictions_per_sample': avg_predictions_per_sample,
            'samples_with_predictions_ratio': samples_with_predictions
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: str
    ):
        """Full training loop with dynamic threshold optimization"""
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
        logger.info(f"Using threshold: {self.current_threshold}")
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"{'='*50}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch, scheduler)
            self.train_history.append(train_metrics)
            
            # Dynamic threshold optimization every 5 epochs
            if self.config.use_dynamic_threshold and epoch % 5 == 0:
                optimal_threshold, optimal_f1 = find_optimal_threshold(
                    self.model,
                    val_loader,
                    self.device,
                    self.config.threshold_search_range,
                    self.config.threshold_search_step
                )
                
                if optimal_f1 > self.best_val_f1 * 0.95:  # Update if within 5% of best
                    self.current_threshold = optimal_threshold
                    logger.info(f"Updated threshold to {self.current_threshold:.3f}")
            
            # Validate with current threshold
            val_metrics = self.validate(val_loader, epoch)
            self.val_history.append(val_metrics)
            
            # Log metrics
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"F1-Macro: {train_metrics['f1_macro']:.4f}, "
                       f"F1-Micro: {train_metrics['f1_micro']:.4f}, "
                       f"Avg Preds/Sample: {train_metrics['avg_predictions_per_sample']:.2f}")
            
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                       f"F1-Macro: {val_metrics['f1_macro']:.4f}, "
                       f"F1-Micro: {val_metrics['f1_micro']:.4f}, "
                       f"Hamming Loss: {val_metrics['hamming_loss']:.4f}, "
                       f"Samples with Preds: {val_metrics['samples_with_predictions_ratio']:.2%}")
            
            # Check for prediction collapse
            if val_metrics['samples_with_predictions_ratio'] < 0.1:
                logger.warning("WARNING: Model is predicting very few positive labels!")
                logger.warning("Consider adjusting loss function or threshold")
            
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
            'config': self.config.__dict__,
            'final_threshold': self.current_threshold
        }
        with open(save_path / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        logger.info(f"\nTraining complete! Best validation F1-Macro: {self.best_val_f1:.4f}")
        logger.info(f"Final threshold: {self.current_threshold:.3f}")
    
    def save_model(self, path: Path, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        model_to_save = self.model.module if isinstance(self.model, DataParallel) else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__,
            'best_val_f1': self.best_val_f1,
            'threshold': self.current_threshold,
            'class_weights': self.class_weights
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def plot_training_curves(self, save_dir: Path):
        """Plot comprehensive training curves"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
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
        
        # Average Predictions per Sample
        ax = axes[2, 0]
        ax.plot(epochs, [h['avg_predictions_per_sample'] for h in self.train_history], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, [h['avg_predictions_per_sample'] for h in self.val_history], 'r-', label='Val', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Avg Predictions')
        ax.set_title('Average Predictions per Sample')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Samples with Predictions Ratio
        ax = axes[2, 1]
        ax.plot(epochs, [h['samples_with_predictions_ratio'] for h in self.val_history], 'orange', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Ratio')
        ax.set_title('Ratio of Samples with At Least One Prediction')
        ax.grid(True, alpha=0.3)
        
        # Threshold Evolution (if dynamic)
        ax = axes[2, 2]
        if 'threshold' in self.val_history[0]:
            ax.plot(epochs, [h.get('threshold', self.config.threshold) for h in self.val_history], 'purple', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Threshold')
            ax.set_title('Prediction Threshold Evolution')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('GoEmotions Multi-Label Training Curves (Fixed)', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()


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
    
    # Get class weights from training dataset
    class_weights = train_dataset.class_weights
    
    # Update config with actual number of labels
    config.num_labels = train_dataset.num_labels
    logger.info(f"Number of emotion labels: {config.num_labels}")
    logger.info(f"Using loss function: {'Focal Loss' if config.use_focal_loss else 'Asymmetric Loss' if config.use_asymmetric_loss else 'Weighted BCE'}")
    
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
    logger.info("Initializing TAN Model with Fixes")
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
    
    model = TopoformerForMultiLabelClassification(topo_config, config.num_labels, config)
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer with class weights
    trainer = MultiLabelTrainer(
        model=model,
        config=config,
        device=device,
        class_weights=class_weights,
        use_data_parallel=use_data_parallel,
        gpu_ids=gpu_ids
    )
    
    # Train model
    logger.info("\n" + "="*60)
    logger.info("Starting Training with Improved Loss and Monitoring")
    logger.info("="*60)
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.num_epochs,
        save_dir=config.save_dir
    )
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Final best F1-Macro: {trainer.best_val_f1:.4f}")
    logger.info(f"Final threshold: {trainer.current_threshold:.3f}")
    logger.info(f"Results saved to: {config.save_dir}")


if __name__ == "__main__":
    main()
