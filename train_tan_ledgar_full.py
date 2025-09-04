"""
Train TAN/Topoformer on LEDGAR Dataset - Fixed for Single-Label Classification with DataParallel
Implements training, validation, and testing for:
1. Classification (100 provision types)
2. Question Answering
3. Retrieval

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
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Import TAN/Topoformer components
try:
    from streamlined_topoformer import TopoformerConfig, TopoformerLayer
    from topoformer_tasks import TopoformerBase
except ImportError:
    print("Warning: Using simplified Topoformer implementation")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopoformerForSingleLabelClassification(nn.Module):
    """TAN/Topoformer for single-label classification (not multi-label)"""
    
    def __init__(self, config: TopoformerConfig, num_labels: int):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embeddings = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        
        # Topoformer layers
        self.layers = nn.ModuleList([
            TopoformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Output layer normalization
        self.output_norm = nn.LayerNorm(config.embed_dim)
        
        # Single-label classifier
        self.classifier = nn.Linear(config.embed_dim, num_labels)
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for single-label classification
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size] - single label per example
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        embeddings = self.embedding_dropout(token_embeds + position_embeds)
        
        # Pass through layers
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Final normalization
        hidden_states = self.output_norm(hidden_states)
        
        # Pool to get sequence representation
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = hidden_states.mean(dim=1)
        
        # Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        outputs = {
            'logits': logits,
            'pooled_output': pooled_output
        }
        
        # Compute loss if labels provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            outputs['loss'] = loss
        
        return outputs


class LEDGARDataset(Dataset):
    """LEDGAR dataset for all tasks"""
    
    def __init__(self, data_path: str, split: str, tokenizer, max_length: int = 512, task: str = 'classification'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task
        self.split = split
        
        # Load data
        data_file = Path(data_path) / f'ledgar_{split}.pt'
        self.data = torch.load(data_file)
        
        if task == 'classification':
            self.texts = self.data['texts']
            self.labels = self.data['labels']
            self.num_classes = self.data.get('num_classes', 100)
            self.provision_types = self.data.get('provision_types', [])
        elif task == 'qa':
            # Load QA data
            qa_file = Path(data_path) / 'ledgar_qa.json'
            with open(qa_file, 'r') as f:
                qa_data = json.load(f)
            # Filter by split if needed
            self.qa_data = qa_data[:len(qa_data)//3] if split == 'train' else qa_data[len(qa_data)//3:]
        elif task == 'retrieval':
            # Load retrieval data
            ret_file = Path(data_path) / 'ledgar_retrieval.json'
            with open(ret_file, 'r') as f:
                ret_data = json.load(f)
            self.retrieval_data = ret_data[:len(ret_data)//3] if split == 'train' else ret_data[len(ret_data)//3:]
    
    def __len__(self):
        if self.task == 'classification':
            return len(self.texts)
        elif self.task == 'qa':
            return len(self.qa_data)
        elif self.task == 'retrieval':
            return len(self.retrieval_data)
    
    def __getitem__(self, idx):
        if self.task == 'classification':
            text = self.texts[idx]
            label = self.labels[idx]
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        
        elif self.task == 'qa':
            item = self.qa_data[idx]
            
            # Encode question and context
            encoding = self.tokenizer(
                item['question'],
                item['context'],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Find answer positions in tokenized text
            answer_text = item['answer']
            # Simplified: use dummy positions for now
            start_position = 10
            end_position = 20
            
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'start_positions': torch.tensor(start_position, dtype=torch.long),
                'end_positions': torch.tensor(end_position, dtype=torch.long)
            }
        
        elif self.task == 'retrieval':
            item = self.retrieval_data[idx]
            
            # Encode query
            query_encoding = self.tokenizer(
                item['query'],
                truncation=True,
                padding='max_length',
                max_length=self.max_length // 2,
                return_tensors='pt'
            )
            
            # For training, we'll use the first relevant doc as positive
            pos_doc = item['relevant_docs'][0] if item['relevant_docs'] else item['candidate_docs'][0]
            
            doc_encoding = self.tokenizer(
                pos_doc,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'query_input_ids': query_encoding['input_ids'].squeeze(0),
                'query_attention_mask': query_encoding['attention_mask'].squeeze(0),
                'doc_input_ids': doc_encoding['input_ids'].squeeze(0),
                'doc_attention_mask': doc_encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(1, dtype=torch.long)  # Positive pair
            }


class TANTrainer:
    """Trainer for TAN/Topoformer models with DataParallel support"""
    
    def __init__(self, 
                 model: nn.Module,
                 task: str,
                 device: torch.device,
                 config: Dict,
                 use_data_parallel: bool = True,
                 gpu_ids: List[int] = None):
        
        self.task = task
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
        self.scheduler = None  # Will be set based on total steps
        
        # History
        self.train_history = []
        self.val_history = []
        
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} - Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass based on task
            if self.task == 'classification':
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs['loss']
                
                # Handle DataParallel loss averaging
                if isinstance(self.model, DataParallel):
                    loss = loss.mean()
                
                # Calculate accuracy
                with torch.no_grad():
                    preds = torch.argmax(outputs['logits'], dim=-1)
                    correct = (preds == batch['labels']).sum().item()
                    total_correct += correct
                    total_samples += batch['labels'].size(0)
            
            elif self.task == 'qa':
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    start_positions=batch['start_positions'],
                    end_positions=batch['end_positions']
                )
                loss = outputs['loss']
                if isinstance(self.model, DataParallel):
                    loss = loss.mean()
            
            elif self.task == 'retrieval':
                outputs = self.model(
                    query_input_ids=batch['query_input_ids'],
                    doc_input_ids=batch['doc_input_ids'],
                    query_attention_mask=batch['query_attention_mask'],
                    doc_attention_mask=batch['doc_attention_mask'],
                    labels=batch.get('labels')
                )
                loss = outputs['loss']
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
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}",
                'acc': f"{total_correct / max(total_samples, 1):.4f}" if self.task == 'classification' else 'N/A'
            })
        
        metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': total_correct / total_samples if self.task == 'classification' else 0
        }
        
        return metrics
    
    def validate(self, val_loader: DataLoader, epoch: int) -> Dict:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} - Validation"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                if self.task == 'classification':
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    loss = outputs['loss']
                    if isinstance(self.model, DataParallel):
                        loss = loss.mean()
                    
                    preds = torch.argmax(outputs['logits'], dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch['labels'].cpu().numpy())
                
                elif self.task == 'qa':
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        start_positions=batch['start_positions'],
                        end_positions=batch['end_positions']
                    )
                    loss = outputs['loss']
                    if isinstance(self.model, DataParallel):
                        loss = loss.mean()
                
                elif self.task == 'retrieval':
                    outputs = self.model(
                        query_input_ids=batch['query_input_ids'],
                        doc_input_ids=batch['doc_input_ids'],
                        query_attention_mask=batch['query_attention_mask'],
                        doc_attention_mask=batch['doc_attention_mask'],
                        labels=batch.get('labels')
                    )
                    loss = outputs['loss']
                    if isinstance(self.model, DataParallel):
                        loss = loss.mean()
                
                total_loss += loss.item()
        
        # Calculate metrics
        metrics = {'loss': total_loss / len(val_loader)}
        
        if self.task == 'classification' and all_preds:
            accuracy = accuracy_score(all_labels, all_preds)
            precision, recall, f1_macro, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='macro', zero_division=0
            )
            f1_micro = f1_score(all_labels, all_preds, average='micro')
            
            metrics.update({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_macro': f1_macro,
                'f1_micro': f1_micro
            })
        
        return metrics
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int,
              save_dir: str):
        """Full training loop"""
        
        # Setup scheduler
        total_steps = len(train_loader) * num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        best_val_metric = 0
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True, parents=True)
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch + 1)
            self.train_history.append(train_metrics)
            
            # Validate
            val_metrics = self.validate(val_loader, epoch + 1)
            self.val_history.append(val_metrics)
            
            # Log metrics
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
            logger.info(f"Val Loss: {val_metrics['loss']:.4f}")
            
            if self.task == 'classification':
                logger.info(f"Train Acc: {train_metrics['accuracy']:.4f}")
                logger.info(f"Val Acc: {val_metrics['accuracy']:.4f}")
                logger.info(f"Val F1-Macro: {val_metrics['f1_macro']:.4f}")
                logger.info(f"Val F1-Micro: {val_metrics['f1_micro']:.4f}")
                
                # Save best model based on F1-macro
                if val_metrics['f1_macro'] > best_val_metric:
                    best_val_metric = val_metrics['f1_macro']
                    self.save_model(save_path / f'best_model_{self.task}.pt', epoch, val_metrics)
                    logger.info(f"Saved best model with F1-macro: {best_val_metric:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_model(save_path / f'checkpoint_epoch_{epoch+1}.pt', epoch, val_metrics)
        
        # Save final model
        self.save_model(save_path / f'final_model_{self.task}.pt', num_epochs, val_metrics)
        
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
        """Plot training and validation curves"""
        fig, axes = plt.subplots(1, 2 if self.task == 'classification' else 1, figsize=(15, 5))
        
        if self.task == 'classification':
            # Loss plot
            ax = axes[0]
            epochs = range(1, len(self.train_history) + 1)
            ax.plot(epochs, [h['loss'] for h in self.train_history], 'b-', label='Train Loss')
            ax.plot(epochs, [h['loss'] for h in self.val_history], 'r-', label='Val Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss')
            ax.legend()
            ax.grid(True)
            
            # Metrics plot
            ax = axes[1]
            ax.plot(epochs, [h['accuracy'] for h in self.train_history], 'b-', label='Train Acc')
            ax.plot(epochs, [h['accuracy'] for h in self.val_history], 'r-', label='Val Acc')
            ax.plot(epochs, [h['f1_macro'] for h in self.val_history], 'g-', label='Val F1-Macro')
            ax.plot(epochs, [h['f1_micro'] for h in self.val_history], 'm-', label='Val F1-Micro')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.set_title('Training and Validation Metrics')
            ax.legend()
            ax.grid(True)
            ax.set_ylim(0, 1)
        else:
            # Just loss plot for other tasks
            ax = axes if self.task != 'classification' else axes[0]
            epochs = range(1, len(self.train_history) + 1)
            ax.plot(epochs, [h['loss'] for h in self.train_history], 'b-', label='Train Loss')
            ax.plot(epochs, [h['loss'] for h in self.val_history], 'r-', label='Val Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training and Validation Loss')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / f'training_curves_{self.task}.png', dpi=300, bbox_inches='tight')
        plt.close()


def test_model(model: nn.Module, test_loader: DataLoader, device: torch.device, task: str) -> Dict:
    """Test the trained model"""
    model.eval()
    
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            if task == 'classification':
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                # Handle DataParallel
                loss = outputs['loss']
                if isinstance(model, DataParallel):
                    loss = loss.mean()
                
                preds = torch.argmax(outputs['logits'], dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                total_loss += loss.item()
            
            elif task == 'qa':
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    start_positions=batch['start_positions'],
                    end_positions=batch['end_positions']
                )
                loss = outputs['loss']
                if isinstance(model, DataParallel):
                    loss = loss.mean()
                total_loss += loss.item()
            
            elif task == 'retrieval':
                outputs = model(
                    query_input_ids=batch['query_input_ids'],
                    doc_input_ids=batch['doc_input_ids'],
                    query_attention_mask=batch['query_attention_mask'],
                    doc_attention_mask=batch['doc_attention_mask'],
                    labels=batch.get('labels')
                )
                loss = outputs['loss']
                if isinstance(model, DataParallel):
                    loss = loss.mean()
                total_loss += loss.item()
    
    # Calculate final metrics
    results = {'loss': total_loss / len(test_loader)}
    
    if task == 'classification' and all_preds:
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        f1_micro = f1_score(all_labels, all_preds, average='micro')
        
        results.update({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'predictions': all_preds,
            'labels': all_labels
        })
        
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(all_labels, all_preds)
        results['confusion_matrix'] = cm
    
    return results


def main():
    """Main training script with DataParallel support"""
    
    # Configuration - Updated for LEDGAR
    config = {
        'data_dir': './processed_data',
        'save_dir': './ledgar_models',
        'batch_size': 32,
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'num_epochs': 10,
        'max_length': 512,
        'ledgar_config': {
            'vocab_size': 50000,
            'embed_dim': 768,
            'num_layers': 6,
            'num_heads': 10,  # Changed from 12 to 10 for LEDGAR
            'num_labels': 100,  # Changed from 28 to 100 for LEDGAR
            'k_neighbors': 32,
            'use_topology': True,
            'max_seq_len': 512,
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
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Results storage
    all_results = {}
    
    # Task 1: Classification
    logger.info("\n" + "="*60)
    logger.info("Task 1: Classification (100 Provision Types)")
    logger.info("="*60)
    
    # Create model
    clf_config = TopoformerConfig(**config['ledgar_config'])
    clf_model = TopoformerForSingleLabelClassification(clf_config, num_labels=100)
    
    # Create datasets
    train_dataset = LEDGARDataset(config['data_dir'], 'train', tokenizer, task='classification')
    val_dataset = LEDGARDataset(config['data_dir'], 'val', tokenizer, task='classification')
    test_dataset = LEDGARDataset(config['data_dir'], 'test', tokenizer, task='classification')
    
    # Create dataloaders with pin_memory for faster GPU transfer
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=8,  # Increased for better GPU utilization
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=8,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=8,
        pin_memory=True
    )
    
    # Train
    trainer = TANTrainer(
        clf_model, 
        'classification', 
        device, 
        config,
        use_data_parallel=use_data_parallel,
        gpu_ids=gpu_ids
    )
    train_history, val_history = trainer.train(
        train_loader, val_loader, 
        config['num_epochs'], 
        os.path.join(config['save_dir'], 'classification')
    )
    
    # Test
    logger.info("\nTesting classification model...")
    test_results = test_model(trainer.model, test_loader, device, 'classification')
    all_results['classification'] = test_results
    
    logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
    logger.info(f"Test F1-Macro: {test_results['f1_macro']:.4f}")
    logger.info(f"Test F1-Micro: {test_results['f1_micro']:.4f}")
    
    # Task 2: Question Answering
    logger.info("\n" + "="*60)
    logger.info("Task 2: Question Answering")
    logger.info("="*60)
    
    # Import QA model
    try:
        from topoformer_tasks import TopoformerForQuestionAnswering
        qa_model = TopoformerForQuestionAnswering(clf_config)
    except:
        logger.warning("QA model not available, skipping...")
        qa_model = None
    
    if qa_model:
        # Create datasets
        train_dataset_qa = LEDGARDataset(config['data_dir'], 'train', tokenizer, task='qa')
        val_dataset_qa = LEDGARDataset(config['data_dir'], 'val', tokenizer, task='qa')
        test_dataset_qa = LEDGARDataset(config['data_dir'], 'test', tokenizer, task='qa')
        
        # Create dataloaders
        train_loader_qa = DataLoader(
            train_dataset_qa, 
            batch_size=config['batch_size'], 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        val_loader_qa = DataLoader(
            val_dataset_qa, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        test_loader_qa = DataLoader(
            test_dataset_qa, 
            batch_size=config['batch_size'], 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        # Train
        qa_trainer = TANTrainer(
            qa_model, 
            'qa', 
            device, 
            config,
            use_data_parallel=use_data_parallel,
            gpu_ids=gpu_ids
        )
        qa_trainer.train(
            train_loader_qa, val_loader_qa,
            config['num_epochs'] // 2,  # Fewer epochs for QA
            os.path.join(config['save_dir'], 'qa')
        )
        
        # Test
        logger.info("\nTesting QA model...")
        qa_test_results = test_model(qa_trainer.model, test_loader_qa, device, 'qa')
        all_results['qa'] = qa_test_results
    
    # Task 3: Retrieval
    logger.info("\n" + "="*60)
    logger.info("Task 3: Retrieval")
    logger.info("="*60)
    
    # Import retrieval model
    try:
        from topoformer_tasks import TopoformerForRetrieval
        ret_model = TopoformerForRetrieval(clf_config, projection_dim=256)
    except:
        logger.warning("Retrieval model not available, skipping...")
        ret_model = None
    
    if ret_model:
        # Create datasets
        train_dataset_ret = LEDGARDataset(config['data_dir'], 'train', tokenizer, task='retrieval')
        val_dataset_ret = LEDGARDataset(config['data_dir'], 'val', tokenizer, task='retrieval')
        test_dataset_ret = LEDGARDataset(config['data_dir'], 'test', tokenizer, task='retrieval')
        
        # Create dataloaders
        train_loader_ret = DataLoader(
            train_dataset_ret, 
            batch_size=config['batch_size']//2,  # Smaller batch for retrieval
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )
        val_loader_ret = DataLoader(
            val_dataset_ret, 
            batch_size=config['batch_size']//2, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        test_loader_ret = DataLoader(
            test_dataset_ret, 
            batch_size=config['batch_size']//2, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True
        )
        
        # Train
        ret_trainer = TANTrainer(
            ret_model, 
            'retrieval', 
            device, 
            config,
            use_data_parallel=use_data_parallel,
            gpu_ids=gpu_ids
        )
        ret_trainer.train(
            train_loader_ret, val_loader_ret,
            config['num_epochs'] // 2,  # Fewer epochs for retrieval
            os.path.join(config['save_dir'], 'retrieval')
        )
        
        # Test
        logger.info("\nTesting retrieval model...")
        ret_test_results = test_model(ret_trainer.model, test_loader_ret, device, 'retrieval')
        all_results['retrieval'] = ret_test_results
    
    # Save all results
    results_path = Path(config['save_dir']) / 'ledgar_tan_results.json'
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for task, results in all_results.items():
            json_results[task] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v 
                for k, v in results.items()
                if k not in ['predictions', 'labels', 'confusion_matrix']
            }
        json.dump(json_results, f, indent=2)
    
    logger.info(f"\nAll results saved to {results_path}")
    
    # Create summary visualization
    create_summary_plot(all_results, Path(config['save_dir']))
    
    return all_results


def create_summary_plot(results: Dict, save_dir: Path):
    """Create summary visualization of all tasks"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Classification results
    ax = axes[0]
    if 'classification' in results:
        metrics = ['accuracy', 'precision', 'recall', 'f1_macro', 'f1_micro']
        values = [results['classification'].get(m, 0) for m in metrics]
        
        bars = ax.bar(range(len(metrics)), values, color=['blue', 'green', 'orange', 'red', 'purple'])
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, rotation=45)
        ax.set_ylabel('Score')
        ax.set_title('Classification Performance (100 Classes)')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # QA results
    ax = axes[1]
    ax.text(0.5, 0.5, 'Question Answering\nResults', ha='center', va='center', fontsize=16)
    ax.set_title('QA Performance')
    ax.axis('off')
    
    # Retrieval results
    ax = axes[2]
    ax.text(0.5, 0.5, 'Retrieval\nResults', ha='center', va='center', fontsize=16)
    ax.set_title('Retrieval Performance')
    ax.axis('off')
    
    plt.suptitle('TAN/Topoformer Performance on LEDGAR Dataset (Multi-GPU)', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_dir / 'ledgar_summary_results.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Set environment variables for better multi-GPU performance
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['TORCH_CUDA_ARCH_LIST'] = '7.0;7.5;8.0;8.6'
    
    main()