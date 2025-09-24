#!/usr/bin/env python3
"""
TPU-Compatible Training Script for TAN and Baselines
Datasets: LEDGAR (100-class legal) and PubMed (long-context medical)
Updates:
- Integrated PyTorch/XLA for TPU training
- Used ParallelLoader for efficient data handling
- Adapted training loop with xm.optimizer_step
- Added xmp.spawn for distributed execution
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
import numpy as np
from pathlib import Path
import json
import time
import logging
import math
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, accuracy_score, precision_recall_fscore_support,
    top_k_accuracy_score, confusion_matrix
)
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# TPU: Import torch_xla libraries
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp


warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================== DATASET CLASSES ==================

class LEDGARDataset(Dataset):
    """LEDGAR dataset with 100 legal document classes - FIXED"""

    def __init__(self, split='train', tokenizer=None, max_length=512, max_samples=50000):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load LEDGAR dataset
        dataset = load_dataset("coastalcph/lex_glue", "ledgar", split=split)

        # Limit samples for training
        if len(dataset) > max_samples:
            indices = np.random.choice(len(dataset), max_samples, replace=False)
            dataset = dataset.select(indices)
            logger.info(f"Limited {split} data to {max_samples} samples")

        # Pre-process and cache
        self.data = []
        logger.info(f"Processing {len(dataset)} samples for {split}...")

        # Track unique labels
        unique_labels = set()

        for item in tqdm(dataset, desc=f"Processing {split}"):
            # FIXED: Use only 'text' field, not 'context' and 'endings'
            if 'text' in item:
                text = item['text']
            else:
                # Fallback for different dataset structures
                text = item.get('premise', '') or item.get('context', '')

            # Ensure text is not empty
            if not text:
                continue

            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            label = item['label']
            unique_labels.add(label)

            self.data.append({
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long)
            })

        self.num_labels = 100  # Fixed to 100 classes for LEDGAR
        self.unique_labels_in_split = len(unique_labels)
        # TPU: Log only on the master process to avoid spam
        if xm.is_master_ordinal():
            logger.info(f"Loaded {len(self.data)} samples")
            logger.info(f"Number of unique labels in {split}: {self.unique_labels_in_split}/{self.num_labels}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PubMedDataset(Dataset):
    """PubMed dataset for long-context classification"""

    def __init__(self, split='train', tokenizer=None, max_length=2048, max_samples=50000):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load PubMed dataset
        dataset = load_dataset("ml4pubmed/pubmed-classification-20k", split=split)

        if len(dataset) > max_samples:
            indices = np.random.choice(len(dataset), max_samples, replace=False)
            dataset = dataset.select(indices)
            logger.info(f"Limited {split} data to {max_samples} samples")

        # Pre-process
        self.data = []
        self.label_map = {}
        unique_labels = set()

        # First pass to get unique labels
        for item in dataset:
            unique_labels.add(item['label'])

        self.label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        self.num_labels = len(self.label_map)

        logger.info(f"Processing {len(dataset)} samples for {split}...")

        for item in tqdm(dataset, desc=f"Processing {split}"):
            text = item.get('text')
            if not text:
                continue

            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )

            self.data.append({
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(self.label_map[item['label']], dtype=torch.long)
            })
        
        # TPU: Log only on the master process
        if xm.is_master_ordinal():
            logger.info(f"Loaded {len(self.data)} samples with {self.num_labels} classes")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ================== TAN/TOPOFORMER IMPLEMENTATION ==================
# NOTE: Model architectures (TAN, TopoformerLayer, etc.) remain unchanged.

class TopologyExtractor(nn.Module):
    """Extract topological features using persistent homology concepts"""
    def __init__(self, embed_dim, k_neighbors=32):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.topo_encoder = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(embed_dim, embed_dim))
    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        queries, keys, values = self.query_proj(hidden_states), self.key_proj(hidden_states), self.value_proj(hidden_states)
        distances = torch.cdist(queries, keys, p=2)
        k = min(self.k_neighbors, seq_len - 1)
        if k <= 0: return self.topo_encoder(hidden_states)
        _, indices = torch.topk(distances, k=k, dim=-1, largest=False)
        batch_indices = torch.arange(batch_size, device=hidden_states.device).view(-1, 1, 1).expand(-1, seq_len, k)
        neighbor_features = values[batch_indices, indices]
        topo_features = neighbor_features.mean(dim=2)
        return self.topo_encoder(topo_features)

class TopoformerLayer(nn.Module):
    """Single Topoformer layer with topology-aware attention"""
    def __init__(self, embed_dim, num_heads, k_neighbors=32):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.topology = TopologyExtractor(embed_dim, k_neighbors)
        self.gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(0.1), nn.Linear(embed_dim * 4, embed_dim))
        self.norm1, self.norm2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x, attention_mask=None):
        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        attn_output, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        topo_features = self.topology(x)
        combined = torch.cat([attn_output, topo_features], dim=-1)
        gate_weights = self.gate(combined)
        gated_output = gate_weights * attn_output + (1 - gate_weights) * topo_features
        x = self.norm1(x + self.dropout(gated_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class TAN(nn.Module):
    """Topological Attention Network - Full Implementation"""
    def __init__(self, num_labels, vocab_size=30522, embed_dim=786, num_layers=6, num_heads=10, k_neighbors=32, max_seq_len=512):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(max_seq_len, embed_dim)
        self.embedding_norm = nn.LayerNorm(embed_dim)
        self.embedding_dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList([TopoformerLayer(embed_dim, num_heads, k_neighbors) for _ in range(num_layers)])
        self.pooler = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh())
        self.classifier = nn.Linear(embed_dim, num_labels)
        self.dropout = nn.Dropout(0.1)
        self._init_weights()
    def _init_weights(self):
        nn.init.normal_(self.embeddings.weight, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        embeddings = self.embeddings(input_ids) + self.position_embeddings(position_ids)
        embeddings = self.embedding_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        hidden_states = embeddings
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (hidden_states * mask_expanded).sum(1)
            sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = hidden_states.mean(1)
        pooled = self.pooler(pooled)
        logits = self.classifier(self.dropout(pooled))
        return {'logits': logits}


# ================== OTHER BASELINE MODELS ==================
# NOTE: Architectures for BERTTopological, HAT, and TopoLM remain unchanged.
class BERTTopological(nn.Module):
    """BERT with topological enhancements"""
    def __init__(self, num_labels, hidden_size=768):
        super().__init__()
        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.topo_extractor = TopologyExtractor(hidden_size, k_neighbors=32)
        self.topo_gate = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid())
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        topo_features = self.topo_extractor(hidden_states)
        pooled, topo_pooled = outputs.pooler_output, topo_features[:, 0]
        combined = torch.cat([pooled, topo_pooled], dim=-1)
        gate = self.topo_gate(combined)
        final_features = gate * pooled + (1 - gate) * topo_pooled
        logits = self.classifier(self.dropout(final_features))
        return {'logits': logits}

class HAT(nn.Module):
    """Hierarchical Attention Transformer"""
    def __init__(self, num_labels, vocab_size=30522, embed_dim=768, num_layers=6):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(2048, embed_dim)
        word_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=3072, dropout=0.1, batch_first=True)
        self.word_encoder = nn.TransformerEncoder(word_layer, num_layers=num_layers//2)
        segment_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=3072, dropout=0.1, batch_first=True)
        self.segment_encoder = nn.TransformerEncoder(segment_layer, num_layers=num_layers//2)
        self.hierarchy_predictor = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, 1))
        self.pooler = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh())
        self.classifier = nn.Linear(embed_dim, num_labels)
        self.dropout = nn.Dropout(0.1)
        nn.init.normal_(self.embeddings.weight, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        embeddings = self.dropout(self.embeddings(input_ids) + self.position_embeddings(position_ids))
        padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        word_output = self.word_encoder(embeddings, src_key_padding_mask=padding_mask)
        segment_boundaries = torch.sigmoid(self.hierarchy_predictor(word_output).squeeze(-1))
        weighted_features = word_output * segment_boundaries.unsqueeze(-1)
        segment_output = self.segment_encoder(weighted_features, src_key_padding_mask=padding_mask)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (segment_output * mask_expanded).sum(1)
            sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = segment_output.mean(dim=1)
        pooled = self.pooler(pooled)
        logits = self.classifier(self.dropout(pooled))
        return {'logits': logits}

class TopoLM(nn.Module):
    """Topology-aware Language Model"""
    def __init__(self, num_labels, vocab_size=30522, embed_dim=768, num_layers=6):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.position_embeddings = nn.Embedding(2048, embed_dim)
        self.embedding_norm = nn.LayerNorm(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=3072, dropout=0.1, batch_first=True)
        self.main_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers-2)
        self.topo_projection = nn.Linear(embed_dim, 384)
        topo_layer = nn.TransformerEncoderLayer(d_model=384, nhead=6, dim_feedforward=1536, dropout=0.1, batch_first=True)
        self.topo_encoder = nn.TransformerEncoder(topo_layer, num_layers=2)
        self.fusion = nn.Sequential(nn.Linear(embed_dim + 384, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(), nn.Dropout(0.1))
        self.pooler = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh())
        self.classifier = nn.Linear(embed_dim, num_labels)
        self.dropout = nn.Dropout(0.1)
        nn.init.normal_(self.embeddings.weight, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        embeddings = self.dropout(self.embedding_norm(self.embeddings(input_ids) + self.position_embeddings(position_ids)))
        padding_mask = ~attention_mask.bool() if attention_mask is not None else None
        main_output = self.main_encoder(embeddings, src_key_padding_mask=padding_mask)
        topo_output = self.topo_encoder(self.topo_projection(main_output), src_key_padding_mask=padding_mask)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            main_pooled = (main_output * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
            topo_pooled = (topo_output * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        else:
            main_pooled, topo_pooled = main_output.mean(dim=1), topo_output.mean(dim=1)
        fused = self.fusion(torch.cat([main_pooled, topo_pooled], dim=-1))
        pooled = self.pooler(fused)
        logits = self.classifier(self.dropout(pooled))
        return {'logits': logits}


# ================== TRAINER CLASS WITH TPU UPDATES ==================

class UnifiedTrainer:
    """Unified trainer for all models with TPU support"""

    def __init__(self, model_name, dataset_name, num_labels, device):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.num_labels = num_labels
        self.device = 'TPU' # TPU: Device is passed in

        # Model configuration
        config = {
            'ledgar': {'max_seq_len': 512, 'embed_dim': 768, 'num_layers': 6, 'num_heads': 10, 'k_neighbors': 32},
            'pubmed': {'max_seq_len': 2048, 'embed_dim': 768, 'num_layers': 6, 'num_heads': 8, 'k_neighbors': 64}
        }[dataset_name]

        # Initialize model
        if model_name == 'TAN':
            self.model = TAN(num_labels=num_labels, embed_dim=config['embed_dim'], num_layers=config['num_layers'], num_heads=config['num_heads'], k_neighbors=config['k_neighbors'], max_seq_len=config['max_seq_len'])
        elif model_name == 'BERT-Topological':
            self.model = BERTTopological(num_labels=num_labels)
        elif model_name == 'HAT':
            self.model = HAT(num_labels=num_labels, embed_dim=config['embed_dim'], num_layers=config['num_layers'])
        elif model_name == 'TopoLM':
            self.model = TopoLM(num_labels=num_labels, embed_dim=config['embed_dim'], num_layers=config['num_layers'])

        self.model = self.model.to(self.device)

        self.history = {
            'train_loss': [], 'val_loss': [],
            'val_f1_macro': [], 'val_accuracy': [],
            'val_top5_accuracy': []
        }
        if xm.is_master_ordinal():
            logger.info(f"{model_name} initialized with {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M parameters")

    def train(self, train_loader, val_loader, num_epochs=7, learning_rate=2e-5):
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        best_f1 = 0

        for epoch in range(num_epochs):
            if xm.is_master_ordinal():
                logger.info(f"\nEpoch {epoch+1}/{num_epochs}")

            # Training
            self.model.train()
            # TPU: Wrap dataloader with ParallelLoader
            para_train_loader = pl.ParallelLoader(train_loader, [self.device])
            train_pbar = tqdm(para_train_loader.per_device_loader(self.device), desc='Training', disable=not xm.is_master_ordinal())
            
            for batch in train_pbar:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs['logits'], labels)
                loss.backward()
                
                # TPU: Use xm.optimizer_step to handle gradient updates
                xm.optimizer_step(optimizer)
                
                if xm.is_master_ordinal():
                    train_pbar.set_postfix({'loss': loss.item()})

            # Validation (run on all cores, but log/save on master)
            val_metrics = self.evaluate(val_loader)
            
            # TPU: Use xm.all_gather to collect metrics from all cores
            val_loss_gathered = xm.all_gather(torch.tensor(val_metrics['loss'], device=self.device))
            val_f1_gathered = xm.all_gather(torch.tensor(val_metrics['f1_macro'], device=self.device))
            
            # Average metrics across all cores
            avg_val_loss = torch.mean(val_loss_gathered).item()
            avg_val_f1 = torch.mean(val_f1_gathered).item()

            if xm.is_master_ordinal():
                self.history['val_loss'].append(avg_val_loss)
                self.history['val_f1_macro'].append(avg_val_f1)
                self.history['val_accuracy'].append(val_metrics['accuracy']) # Acc and top5 are calculated on full data
                self.history['val_top5_accuracy'].append(val_metrics.get('top5_accuracy', 0))

                logger.info(f"Val Loss: {avg_val_loss:.4f}")
                logger.info(f"Val F1-Macro: {avg_val_f1:.4f}")
                logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")

                if avg_val_f1 > best_f1:
                    best_f1 = avg_val_f1
                    # TPU: Use xm.save for saving models
                    self.save_model(f'best_{self.model_name}_{self.dataset_name}.pt')
                    logger.info(f"Saved best model with F1-Macro: {best_f1:.4f}")
        return best_f1

    def evaluate(self, dataloader):
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        total_loss = 0
        criterion = nn.CrossEntropyLoss()
        
        # TPU: Wrap dataloader with ParallelLoader for evaluation
        para_eval_loader = pl.ParallelLoader(dataloader, [self.device])
        eval_pbar = tqdm(para_eval_loader.per_device_loader(self.device), desc='Evaluating', disable=not xm.is_master_ordinal())

        with torch.no_grad():
            for batch in eval_pbar:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs['logits'], labels)
                total_loss += loss.item()

                probs = F.softmax(outputs['logits'], dim=-1)
                preds = torch.argmax(outputs['logits'], 1)
                
                all_probs.append(xm.mesh_reduce('all_probs', probs, torch.cat))
                all_preds.append(xm.mesh_reduce('all_preds', preds, torch.cat))
                all_labels.append(xm.mesh_reduce('all_labels', labels, torch.cat))

        # Combine results from all cores
        all_probs = torch.cat([p.cpu() for p in all_probs]).numpy()
        all_preds = torch.cat([p.cpu() for p in all_preds]).numpy()
        all_labels = torch.cat([l.cpu() for l in all_labels]).numpy()

        metrics = {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0)
        }
        if all_probs.shape[1] >= 5:
            all_possible_labels = list(range(self.num_labels))
            metrics['top5_accuracy'] = top_k_accuracy_score(all_labels, all_probs, k=5, labels=all_possible_labels)
        
        return metrics

    def save_model(self, path):
        # TPU: Use xm.save and save only on the master process
        if xm.is_master_ordinal():
            xm.save({
                'model_state_dict': self.model.state_dict(),
                'model_name': self.model_name,
                'dataset_name': self.dataset_name,
                'num_labels': self.num_labels,
                'history': self.history
            }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)


# ================== MAIN TRAINING SCRIPT ==================

def _mp_fn(index, flags):
    """Main training function for a single TPU core."""
    # TPU: Get the XLA device for the current process
    device = xm.xla_device()
    
    config = flags['config']
    dataset_name = flags['pubmed']
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Load datasets
    if dataset_name == 'ledgar':
        train_dataset = LEDGARDataset('train', tokenizer, config['max_seq_len'], max_samples=50000)
        val_dataset = LEDGARDataset('validation', tokenizer, config['max_seq_len'], max_samples=5000)
        test_dataset = LEDGARDataset('test', tokenizer, config['max_seq_len'], max_samples=5000)
        num_labels = 100
    else: # pubmed
        train_dataset = PubMedDataset('train', tokenizer, config['max_seq_len'], max_samples=15000)
        val_dataset = PubMedDataset('validation', tokenizer, config['max_seq_len'], max_samples=2500)
        test_dataset = PubMedDataset('test', tokenizer, config['max_seq_len'], max_samples=2500)
        num_labels = train_dataset.num_labels

    # TPU: Use DistributedSampler for DataLoaders
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=False)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=train_sampler, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], sampler=val_sampler, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], sampler=test_sampler, num_workers=1)
    
    results_file = flags['results_file']
    results = flags['results']

    for model_name in flags['models']:
        model_file = f'best_{model_name}_{dataset_name}.pt'
        
        if xm.is_master_ordinal():
            logger.info(f"\nTraining {model_name} on {dataset_name}...")

        # Initialize trainer
        trainer = UnifiedTrainer(model_name, dataset_name, num_labels, device)

        # Train
        best_f1 = trainer.train(train_loader, val_loader, num_epochs=config['num_epochs'], learning_rate=config['learning_rate'])

        # Load best model and evaluate on test set
        # TPU: Ensure all processes are synchronized before loading/evaluating
        xm.rendezvous('training_finished')
        
        if os.path.exists(model_file):
            trainer.load_model(model_file)
            test_metrics = trainer.evaluate(test_loader)
        else:
            if xm.is_master_ordinal():
                 logger.warning(f"Model file {model_file} not found. Skipping evaluation.")
            test_metrics = {'f1_macro': 0, 'accuracy': 0, 'top5_accuracy': 0}

        if xm.is_master_ordinal():
            if dataset_name not in results:
                results[dataset_name] = {}
            results[dataset_name][model_name] = {
                'val_best_f1': best_f1,
                'test_f1_macro': test_metrics['f1_macro'],
                'test_accuracy': test_metrics['accuracy'],
                'test_top5_accuracy': test_metrics.get('top5_accuracy', 0),
                'history': trainer.history
            }
            logger.info(f"\n{model_name} Test Results:")
            logger.info(f"  F1-Macro: {test_metrics['f1_macro']:.4f}")
            logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")

            # Save results progress
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Progress saved to {results_file}")

def main():
    logger.info("="*80)
    logger.info("TPU TRAINING SCRIPT")
    logger.info("Datasets: LEDGAR & PubMed")
    logger.info("="*80)

    configs = {
        'ledgar': {'max_seq_len': 512, 'batch_size': 16, 'num_epochs': 7, 'learning_rate': 2e-5},
        'pubmed': {'max_seq_len': 2048, 'batch_size': 16, 'num_epochs': 7, 'learning_rate': 2e-5}
    }
    models = ['TAN', 'BERT-Topological', 'HAT', 'TopoLM']
    results = {}
    results_file = 'complex_datasets_results_tpu.json'

    if os.path.exists(results_file):
        logger.info("Loading existing results...")
        with open(results_file, 'r') as f:
            results = json.load(f)

    for dataset_name in ['pubmed']: # Or ['ledgar', 'pubmed']
        flags = {
            'config': configs[dataset_name],
            'dataset_name': dataset_name,
            'models': models,
            'results': results,
            'results_file': results_file
        }
        # TPU: Use xmp.spawn to start the training on all TPU cores
        xmp.spawn(_mp_fn, args=(flags,), start_method='spawn')

    logger.info("\nTraining complete! Results saved to complex_datasets_results_tpu.json")


if __name__ == "__main__":
    main()
