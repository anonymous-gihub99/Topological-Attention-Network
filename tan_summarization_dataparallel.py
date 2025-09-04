"""
TAN/Topoformer for Text Summarization on BookSum Dataset
Implements abstractive summarization using Topoformer architecture
Modified for multi-GPU training with DataParallel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Import TAN/Topoformer components
try:
    from streamlined_topoformer import TopoformerConfig, TopoformerLayer
except ImportError:
    print("Warning: Using simplified Topoformer implementation")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopoformerForSummarization(nn.Module):
    """TAN/Topoformer adapted for abstractive summarization"""
    
    def __init__(self, config: TopoformerConfig):
        super().__init__()
        self.config = config
        
        # Shared embeddings
        self.shared_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            TopoformerLayer(config) for _ in range(config.num_layers)
        ])
        self.encoder_norm = nn.LayerNorm(config.embed_dim)
        
        # Decoder with cross-attention
        self.decoder_layers = nn.ModuleList([
            self._create_decoder_layer(config) for _ in range(config.num_layers)
        ])
        self.decoder_norm = nn.LayerNorm(config.embed_dim)
        
        # Output projection
        self.output_projection = nn.Linear(config.embed_dim, config.vocab_size)
        
        # Tie embeddings with output projection
        self.output_projection.weight = self.shared_embeddings.weight
        
        # Initialize
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
    
    def _create_decoder_layer(self, config):
        """Create a decoder layer with self-attention and cross-attention"""
        return nn.ModuleDict({
            'self_attn': nn.MultiheadAttention(
                config.embed_dim, 
                config.num_heads,
                dropout=config.dropout,
                batch_first=True
            ),
            'self_attn_norm': nn.LayerNorm(config.embed_dim),
            'cross_attn': nn.MultiheadAttention(
                config.embed_dim,
                config.num_heads,
                dropout=config.dropout,
                batch_first=True
            ),
            'cross_attn_norm': nn.LayerNorm(config.embed_dim),
            'ffn': nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim * 4),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.embed_dim * 4, config.embed_dim),
                nn.Dropout(config.dropout)
            ),
            'ffn_norm': nn.LayerNorm(config.embed_dim)
        })
    
    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode source text"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        embeddings = self.shared_embeddings(input_ids) + self.position_embeddings(position_ids)
        embeddings = self.embedding_dropout(embeddings)
        
        # Encode through Topoformer layers
        hidden_states = embeddings
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        encoded = self.encoder_norm(hidden_states)
        return encoded
    
    def decode(self, 
               decoder_input_ids: torch.Tensor,
               encoder_hidden_states: torch.Tensor,
               decoder_attention_mask: Optional[torch.Tensor] = None,
               encoder_attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode with cross-attention to encoder states"""
        batch_size, seq_len = decoder_input_ids.shape
        device = decoder_input_ids.device
        
        # Embeddings
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        embeddings = self.shared_embeddings(decoder_input_ids) + self.position_embeddings(position_ids)
        embeddings = self.embedding_dropout(embeddings)
        
        # Create causal mask for decoder self-attention
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        
        # Decode through layers
        hidden_states = embeddings
        for layer in self.decoder_layers:
            # Self-attention with causal mask
            residual = hidden_states
            hidden_states = layer['self_attn_norm'](hidden_states)
            attn_output, _ = layer['self_attn'](
                hidden_states, hidden_states, hidden_states,
                attn_mask=causal_mask,
                key_padding_mask=~decoder_attention_mask.bool() if decoder_attention_mask is not None else None
            )
            hidden_states = residual + attn_output
            
            # Cross-attention to encoder
            residual = hidden_states
            hidden_states = layer['cross_attn_norm'](hidden_states)
            cross_output, _ = layer['cross_attn'](
                hidden_states, encoder_hidden_states, encoder_hidden_states,
                key_padding_mask=~encoder_attention_mask.bool() if encoder_attention_mask is not None else None
            )
            hidden_states = residual + cross_output
            
            # FFN
            residual = hidden_states
            hidden_states = layer['ffn_norm'](hidden_states)
            hidden_states = residual + layer['ffn'](hidden_states)
        
        decoded = self.decoder_norm(hidden_states)
        return decoded
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                decoder_input_ids: Optional[torch.Tensor] = None,
                decoder_attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training
        
        Args:
            input_ids: Source text token IDs [batch_size, src_len]
            attention_mask: Source attention mask
            decoder_input_ids: Target text token IDs (shifted right) [batch_size, tgt_len]
            decoder_attention_mask: Target attention mask
            labels: Target text token IDs for loss computation [batch_size, tgt_len]
        """
        # Encode source
        encoder_hidden_states = self.encode(input_ids, attention_mask)
        
        # Decode
        if decoder_input_ids is not None:
            decoder_outputs = self.decode(
                decoder_input_ids,
                encoder_hidden_states,
                decoder_attention_mask,
                attention_mask
            )
            
            # Project to vocabulary
            logits = self.output_projection(decoder_outputs)
            
            outputs = {'logits': logits}
            
            # Calculate loss if labels provided
            if labels is not None:
                loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))
                outputs['loss'] = loss
            
            return outputs
        else:
            # Inference mode - will need to implement generation
            return {'encoder_hidden_states': encoder_hidden_states}
    
    def generate(self,
                 input_ids: torch.Tensor,
                 attention_mask: Optional[torch.Tensor] = None,
                 max_length: int = 150,
                 min_length: int = 30,
                 num_beams: int = 4,
                 length_penalty: float = 2.0,
                 no_repeat_ngram_size: int = 3) -> torch.Tensor:
        """Generate summary using beam search"""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Encode source
        encoder_hidden_states = self.encode(input_ids, attention_mask)
        
        # Initialize with BOS token (assume 101 is [CLS]/BOS)
        generated = torch.full((batch_size, 1), 101, dtype=torch.long, device=device)
        
        # Simple greedy generation (beam search would be more complex)
        for _ in range(max_length):
            # Decode current sequence
            decoder_outputs = self.decode(
                generated,
                encoder_hidden_states,
                None,
                attention_mask
            )
            
            # Get next token logits
            next_token_logits = self.output_projection(decoder_outputs[:, -1, :])
            
            # Apply no_repeat_ngram constraint
            if no_repeat_ngram_size > 0 and generated.shape[1] >= no_repeat_ngram_size:
                # Simple n-gram blocking
                for i in range(batch_size):
                    if generated.shape[1] >= no_repeat_ngram_size:
                        ngram = generated[i, -(no_repeat_ngram_size-1):].tolist()
                        # Find matching n-grams
                        for j in range(generated.shape[1] - no_repeat_ngram_size + 1):
                            if generated[i, j:j+no_repeat_ngram_size-1].tolist() == ngram:
                                # Block the next token
                                next_idx = generated[i, j+no_repeat_ngram_size-1].item()
                                next_token_logits[i, next_idx] = -float('inf')
            
            # Greedy selection
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tokens], dim=-1)
            
            # Stop if all sequences have EOS (assume 102 is [SEP]/EOS)
            if (next_tokens == 102).all():
                break
            
            # Stop at max length
            if generated.shape[1] >= max_length:
                break
        
        return generated


class BookSumDataset(Dataset):
    """BookSum dataset for summarization"""
    
    def __init__(self, 
                 split: str,
                 tokenizer,
                 max_source_length: int = 1024,
                 max_target_length: int = 256,
                 num_samples: Optional[int] = None):
        
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        # Load BookSum dataset
        logger.info(f"Loading BookSum {split} split...")
        dataset = load_dataset("kmfoda/booksum", split=split)
        
        # Limit samples if specified
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        self.data = []
        for item in tqdm(dataset, desc=f"Processing {split} data"):
            # BookSum has 'chapter' and 'summary' fields
            self.data.append({
                'source': item['chapter'][:5000],  # Limit source length
                'target': item['summary'][:1000]   # Limit target length
            })
        
        logger.info(f"Loaded {len(self.data)} samples for {split}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize source
        source_encoding = self.tokenizer(
            item['source'],
            truncation=True,
            padding='max_length',
            max_length=self.max_source_length,
            return_tensors='pt'
        )
        
        # Tokenize target
        with self.tokenizer.as_target_tokenizer():
            target_encoding = self.tokenizer(
                item['target'],
                truncation=True,
                padding='max_length',
                max_length=self.max_target_length,
                return_tensors='pt'
            )
        
        # Prepare decoder input (shift target right)
        decoder_input_ids = target_encoding['input_ids'].squeeze()
        labels = decoder_input_ids.clone()
        
        # Shift decoder input (add BOS, remove last token)
        decoder_input_ids[1:] = labels[:-1]
        decoder_input_ids[0] = self.tokenizer.cls_token_id  # BOS token
        
        # Set padding tokens in labels to -100 (ignored by loss)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'decoder_input_ids': decoder_input_ids,
            'decoder_attention_mask': target_encoding['attention_mask'].squeeze(),
            'labels': labels
        }


class SummarizationTrainer:
    """Trainer for summarization models with DataParallel support"""
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 config: Dict,
                 use_data_parallel: bool = True,
                 gpu_ids: List[int] = None):
        
        self.device = device
        self.config = config
        self.use_data_parallel = use_data_parallel
        
        # Setup DataParallel if requested
        if use_data_parallel and torch.cuda.device_count() > 1:
            if gpu_ids is None:
                gpu_ids = list(range(torch.cuda.device_count()))
            logger.info(f"Using DataParallel on GPUs: {gpu_ids}")
            model = model.to(device)
            self.model = DataParallel(model, device_ids=gpu_ids)
        else:
            self.model = model.to(device)
            logger.info(f"Using single GPU/CPU: {device}")
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = None
        
        # ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # History
        self.train_history = []
        self.val_history = []
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} - Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                decoder_input_ids=batch['decoder_input_ids'],
                decoder_attention_mask=batch['decoder_attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs['loss']
            
            # Handle DataParallel loss averaging
            if isinstance(self.model, DataParallel):
                loss = loss.mean()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
            })
        
        return {'loss': total_loss / len(train_loader)}
    
    def validate(self, val_loader: DataLoader, epoch: int, tokenizer) -> Dict:
        """Validate model and compute ROUGE scores"""
        self.model.eval()
        total_loss = 0
        
        # For ROUGE computation
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} - Validation"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass for loss
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    decoder_input_ids=batch['decoder_input_ids'],
                    decoder_attention_mask=batch['decoder_attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs['loss']
                if isinstance(self.model, DataParallel):
                    loss = loss.mean()
                
                total_loss += loss.item()
                
                # Generate summaries for ROUGE
                if len(all_predictions) < 100:  # Limit for speed
                    # Get the model for generation (unwrap DataParallel if needed)
                    model_for_gen = self.model.module if isinstance(self.model, DataParallel) else self.model
                    
                    generated = model_for_gen.generate(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        max_length=150
                    )
                    
                    # Decode predictions and references
                    for i in range(generated.shape[0]):
                        pred = tokenizer.decode(generated[i], skip_special_tokens=True)
                        ref = tokenizer.decode(
                            batch['labels'][i][batch['labels'][i] != -100], 
                            skip_special_tokens=True
                        )
                        all_predictions.append(pred)
                        all_references.append(ref)
        
        # Calculate ROUGE scores
        rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
        
        if all_predictions:
            for pred, ref in zip(all_predictions, all_references):
                scores = self.rouge_scorer.score(ref, pred)
                for key in rouge_scores:
                    rouge_scores[key] += scores[key].fmeasure
            
            # Average ROUGE scores
            num_samples = len(all_predictions)
            for key in rouge_scores:
                rouge_scores[key] /= num_samples
        
        metrics = {
            'loss': total_loss / len(val_loader),
            **rouge_scores
        }
        
        return metrics
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int,
              save_dir: str,
              tokenizer):
        """Full training loop"""
        
        # Setup scheduler
        total_steps = len(train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        best_rouge_l = 0
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch + 1)
            self.train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch + 1, tokenizer)
            self.val_history.append(val_metrics)
            
            # Log metrics
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            logger.info(f"Val ROUGE-1: {val_metrics['rouge1']:.4f}")
            logger.info(f"Val ROUGE-2: {val_metrics['rouge2']:.4f}")
            logger.info(f"Val ROUGE-L: {val_metrics['rougeL']:.4f}")
            
            # Save best model based on ROUGE-L
            if val_metrics['rougeL'] > best_rouge_l:
                best_rouge_l = val_metrics['rougeL']
                self.save_model(save_path / 'best_model_summarization.pt', epoch, val_metrics)
                logger.info(f"Saved best model with ROUGE-L: {best_rouge_l:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_model(save_path / f'checkpoint_epoch_{epoch+1}.pt', epoch, val_metrics)
        
        # Plot training curves
        self.plot_training_curves(save_path)
        
        return self.train_history, self.val_history
    
    def save_model(self, path: Path, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        # Handle DataParallel wrapper
        model_to_save = self.model.module if isinstance(self.model, DataParallel) else self.model
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }, path)
    
    def plot_training_curves(self, save_dir: Path):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.train_history) + 1)
        
        # Loss plot
        ax = axes[0]
        ax.plot(epochs, [h['loss'] for h in self.train_history], 'b-', label='Train Loss')
        ax.plot(epochs, [h['loss'] for h in self.val_history], 'r-', label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True)
        
        # ROUGE scores plot
        ax = axes[1]
        ax.plot(epochs, [h['rouge1'] for h in self.val_history], 'g-', label='ROUGE-1')
        ax.plot(epochs, [h['rouge2'] for h in self.val_history], 'b-', label='ROUGE-2')
        ax.plot(epochs, [h['rougeL'] for h in self.val_history], 'r-', label='ROUGE-L')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('ROUGE Score')
        ax.set_title('Validation ROUGE Scores')
        ax.legend()
        ax.grid(True)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_curves_summarization.png', dpi=300, bbox_inches='tight')
        plt.close()


def test_summarization(model: nn.Module, test_loader: DataLoader, device: torch.device, tokenizer) -> Dict:
    """Test the summarization model"""
    model.eval()
    
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    all_rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    generated_summaries = []
    reference_summaries = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Generate summaries
            # Handle DataParallel for generation
            model_for_gen = model.module if isinstance(model, DataParallel) else model
            
            generated = model_for_gen.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=150,
                min_length=30
            )
            
            # Decode and compute ROUGE
            for i in range(generated.shape[0]):
                # Decode prediction
                pred = tokenizer.decode(generated[i], skip_special_tokens=True)
                
                # Decode reference
                ref_ids = batch['labels'][i]
                ref_ids = ref_ids[ref_ids != -100]  # Remove padding
                ref = tokenizer.decode(ref_ids, skip_special_tokens=True)
                
                # Store for analysis
                generated_summaries.append(pred)
                reference_summaries.append(ref)
                
                # Compute ROUGE scores
                scores = rouge_scorer_obj.score(ref, pred)
                for key in all_rouge_scores:
                    all_rouge_scores[key].append(scores[key].fmeasure)
    
    # Average ROUGE scores
    results = {
        'rouge1': np.mean(all_rouge_scores['rouge1']),
        'rouge2': np.mean(all_rouge_scores['rouge2']),
        'rougeL': np.mean(all_rouge_scores['rougeL']),
        'rouge1_std': np.std(all_rouge_scores['rouge1']),
        'rouge2_std': np.std(all_rouge_scores['rouge2']),
        'rougeL_std': np.std(all_rouge_scores['rougeL']),
        'generated_summaries': generated_summaries[:10],  # Save some examples
        'reference_summaries': reference_summaries[:10]
    }
    
    return results


def main():
    """Main training script for summarization with DataParallel support"""
    
    # Configuration
    config = {
        'save_dir': './booksum_models',
        'batch_size': 8,  # Increased batch size for multi-GPU
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'num_epochs': 10,
        'max_source_length': 1024,
        'max_target_length': 256,
        'num_train_samples': 5000,  # Limit for faster training
        'num_val_samples': 500,
        'num_test_samples': 500,
        'topoformer_config': {
            'vocab_size': 50265,  # BART vocab size
            'embed_dim': 768,
            'num_layers': 6,
            'num_heads': 12,
            'max_seq_len': 1024,
            'k_neighbors': 32,
            'use_topology': True,
            'dropout': 0.1
        }
    }
    
    # Setup device and GPUs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()
    logger.info(f"Using device: {device}")
    logger.info(f"Number of GPUs available: {num_gpus}")
    
    # Specify GPU IDs to use (for 4 GPUs)
    gpu_ids = list(range(min(4, num_gpus)))  # Use up to 4 GPUs
    use_data_parallel = num_gpus > 1
    
    # Initialize tokenizer (using BART tokenizer for better summarization)
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    
    # Update vocab size
    config['topoformer_config']['vocab_size'] = len(tokenizer)
    
    logger.info("\n" + "="*60)
    logger.info("Text Summarization with TAN/Topoformer on BookSum")
    logger.info("="*60)
    
    # Create model
    topo_config = TopoformerConfig(**config['topoformer_config'])
    model = TopoformerForSummarization(topo_config)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create datasets
    logger.info("\nLoading BookSum dataset...")
    train_dataset = BookSumDataset(
        'train', 
        tokenizer,
        config['max_source_length'],
        config['max_target_length'],
        config['num_train_samples']
    )
    
    val_dataset = BookSumDataset(
        'validation',
        tokenizer,
        config['max_source_length'],
        config['max_target_length'],
        config['num_val_samples']
    )
    
    test_dataset = BookSumDataset(
        'test',
        tokenizer,
        config['max_source_length'],
        config['max_target_length'],
        config['num_test_samples']
    )
    
    # Create dataloaders with pin_memory for faster GPU transfer
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Train
    trainer = SummarizationTrainer(
        model, 
        device, 
        config,
        use_data_parallel=use_data_parallel,
        gpu_ids=gpu_ids
    )
    train_history, val_history = trainer.train(
        train_loader,
        val_loader,
        config['num_epochs'],
        config['save_dir'],
        tokenizer
    )
    
    # Test
    logger.info("\n" + "="*60)
    logger.info("Testing summarization model...")
    logger.info("="*60)
    
    test_results = test_summarization(trainer.model, test_loader, device, tokenizer)
    
    logger.info(f"\nTest Results:")
    logger.info(f"ROUGE-1: {test_results['rouge1']:.4f} Â± {test_results['rouge1_std']:.4f}")
    logger.info(f"ROUGE-2: {test_results['rouge2']:.4f} Â± {test_results['rouge2_std']:.4f}")
    logger.info(f"ROUGE-L: {test_results['rougeL']:.4f} Â± {test_results['rougeL_std']:.4f}")
    
    # Save results
    results_path = Path(config['save_dir']) / 'summarization_results.json'
    with open(results_path, 'w') as f:
        json_results = {
            'rouge1': test_results['rouge1'],
            'rouge2': test_results['rouge2'],
            'rougeL': test_results['rougeL'],
            'rouge1_std': test_results['rouge1_std'],
            'rouge2_std': test_results['rouge2_std'],
            'rougeL_std': test_results['rougeL_std']
        }
        json.dump(json_results, f, indent=2)
    
    # Save some example summaries
    examples_path = Path(config['save_dir']) / 'example_summaries.txt'
    with open(examples_path, 'w') as f:
        for i in range(min(5, len(test_results['generated_summaries']))):
            f.write(f"Example {i+1}:\n")
            f.write(f"Reference: {test_results['reference_summaries'][i]}\n")
            f.write(f"Generated: {test_results['generated_summaries'][i]}\n")
            f.write("-" * 80 + "\n\n")
    
    logger.info(f"\nResults saved to {config['save_dir']}")
    
    return test_results


if __name__ == "__main__":
    # Set environment variables for better multi-GPU performance
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0;7.5;8.0;8.6'
    
    main()