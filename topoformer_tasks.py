"""
Topoformer Task-Specific Models
Implements Classification, Q&A, and Information Retrieval using Topoformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from streamlined_topoformer import TopoformerConfig, TopoformerLayer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopoformerBase(nn.Module):
    """Base Topoformer model with shared architecture"""
    
    def __init__(self, config: TopoformerConfig):
        super().__init__()
        self.config = config
        
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
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings with position encoding"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        embeddings = self.embedding_dropout(token_embeds + position_embeds)
        
        return embeddings
    
    def forward_encoder(self, embeddings: torch.Tensor, 
                       attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pass through Topoformer layers"""
        hidden_states = embeddings
        
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # Final normalization
        hidden_states = self.output_norm(hidden_states)
        
        return hidden_states


class TopoformerForMultiLabelClassification(TopoformerBase):
    """Topoformer for multi-label hierarchical classification"""
    
    def __init__(self, config: TopoformerConfig, num_labels: int,
                 label_hierarchy: Optional[Dict] = None):
        super().__init__(config)
        
        self.num_labels = num_labels
        self.label_hierarchy = label_hierarchy
        
        # Classification heads
        if label_hierarchy:
            # Hierarchical classification
            self.num_groups = len(label_hierarchy)
            self.group_classifier = nn.Linear(config.embed_dim, self.num_groups)
            self.label_classifiers = nn.ModuleDict()
            
            for group_name, group_labels in label_hierarchy.items():
                if isinstance(group_labels, dict):
                    num_group_labels = len(group_labels)
                elif isinstance(group_labels, list):
                    num_group_labels = len(group_labels)
                else:
                    num_group_labels = group_labels
                    
                self.label_classifiers[group_name] = nn.Linear(
                    config.embed_dim + self.num_groups,
                    num_group_labels
                )
        else:
            # Flat multi-label classification
            self.classifier = nn.Linear(config.embed_dim, num_labels)
        
        # Loss function for multi-label
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # Get embeddings and encode
        embeddings = self.forward_embeddings(input_ids)
        hidden_states = self.forward_encoder(embeddings, attention_mask)
        
        # Pool to get sequence representation
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = hidden_states.mean(dim=1)
        
        outputs = {'pooled_output': pooled_output}
        
        if self.label_hierarchy:
            # Hierarchical classification
            group_logits = self.group_classifier(pooled_output)
            outputs['group_logits'] = group_logits
            
            # Group-conditioned label predictions
            group_probs = F.softmax(group_logits, dim=-1)
            group_features = torch.cat([pooled_output, group_probs], dim=-1)
            
            label_logits = {}
            for group_name, classifier in self.label_classifiers.items():
                label_logits[group_name] = classifier(group_features)
            
            outputs['label_logits'] = label_logits
            
            # Combine all logits for loss computation
            if labels is not None:
                # Assuming labels are provided as flat multi-label vector
                # This is a simplified approach - in practice, you'd map hierarchical predictions
                all_logits = []
                for group_name in sorted(self.label_classifiers.keys()):
                    all_logits.append(label_logits[group_name])
                combined_logits = torch.cat(all_logits, dim=-1)
                
                loss = self.loss_fn(combined_logits[:, :self.num_labels], labels.float())
                outputs['loss'] = loss
        else:
            # Flat classification
            logits = self.classifier(pooled_output)
            outputs['logits'] = logits
            
            if labels is not None:
                loss = self.loss_fn(logits, labels.float())
                outputs['loss'] = loss
        
        return outputs


class TopoformerForQuestionAnswering(TopoformerBase):
    """Topoformer for extractive question answering"""
    
    def __init__(self, config: TopoformerConfig):
        super().__init__(config)
        
        # QA outputs
        self.qa_outputs = nn.Linear(config.embed_dim, 2)  # start and end logits
        
        # Optional: Question-aware pooling
        self.question_pooling = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim),
            nn.Tanh()
        )
        
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                start_positions: Optional[torch.Tensor] = None,
                end_positions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        # Get embeddings and encode
        embeddings = self.forward_embeddings(input_ids)
        sequence_output = self.forward_encoder(embeddings, attention_mask)
        
        # QA logits
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        outputs = {
            'start_logits': start_logits,
            'end_logits': end_logits,
            'hidden_states': sequence_output
        }
        
        # Calculate loss if positions provided
        if start_positions is not None and end_positions is not None:
            # Sometimes the start/end positions are outside our model inputs
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)
            
            loss_fn = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fn(start_logits, start_positions)
            end_loss = loss_fn(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            
            outputs['loss'] = total_loss
        
        return outputs


class TopoformerForRetrieval(TopoformerBase):
    """Topoformer for information retrieval with dual encoder"""
    
    def __init__(self, config: TopoformerConfig, 
                 projection_dim: int = 256,
                 use_pooler: bool = True):
        super().__init__(config)
        
        self.projection_dim = projection_dim
        self.use_pooler = use_pooler
        
        # Pooler for better representations
        if use_pooler:
            self.pooler = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.Tanh()
            )
        
        # Projection for retrieval (smaller dimension for efficiency)
        self.projection = nn.Linear(config.embed_dim, projection_dim)
        
        # Temperature parameter for contrastive loss
        self.temperature = nn.Parameter(torch.tensor(0.07))
        
    def forward_document(self, input_ids: torch.Tensor,
                        attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode document"""
        embeddings = self.forward_embeddings(input_ids)
        hidden_states = self.forward_encoder(embeddings, attention_mask)
        
        # Pool
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = hidden_states.mean(dim=1)
        
        # Apply pooler if available
        if self.use_pooler:
            pooled_output = self.pooler(pooled_output)
        
        # Project to retrieval space
        doc_embedding = self.projection(pooled_output)
        doc_embedding = F.normalize(doc_embedding, p=2, dim=-1)
        
        return doc_embedding
    
    def forward_query(self, input_ids: torch.Tensor,
                     attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode query (can be different from document encoding)"""
        # For now, use same encoding as documents
        return self.forward_document(input_ids, attention_mask)
    
    def compute_similarity(self, query_embeddings: torch.Tensor,
                          doc_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute similarity scores between queries and documents"""
        # Cosine similarity with temperature scaling
        similarity = torch.matmul(query_embeddings, doc_embeddings.T) / self.temperature
        return similarity
    
    def forward(self, query_input_ids: torch.Tensor,
                doc_input_ids: torch.Tensor,
                query_attention_mask: Optional[torch.Tensor] = None,
                doc_attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training with contrastive loss
        
        Args:
            query_input_ids: [batch_size, query_len]
            doc_input_ids: [batch_size, doc_len]
            labels: [batch_size] - indices of positive documents
        """
        # Encode queries and documents
        query_embeddings = self.forward_query(query_input_ids, query_attention_mask)
        doc_embeddings = self.forward_document(doc_input_ids, doc_attention_mask)
        
        # Compute similarities
        similarities = self.compute_similarity(query_embeddings, doc_embeddings)
        
        outputs = {
            'query_embeddings': query_embeddings,
            'doc_embeddings': doc_embeddings,
            'similarities': similarities
        }
        
        # Compute contrastive loss if labels provided
        if labels is not None:
            # In-batch negatives: each query's positive is its corresponding document
            # All other documents in the batch are negatives
            batch_size = query_embeddings.size(0)
            
            if labels is None:
                # Default: diagonal elements are positives
                labels = torch.arange(batch_size, device=query_embeddings.device)
            
            # Cross-entropy loss over similarities
            loss = F.cross_entropy(similarities, labels)
            outputs['loss'] = loss
        
        return outputs


class TopoformerForSequenceRanking(TopoformerBase):
    """Topoformer for ranking/reranking tasks"""
    
    def __init__(self, config: TopoformerConfig):
        super().__init__(config)
        
        # Ranking head
        self.rank_classifier = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim // 2, 1)
        )
        
    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for ranking
        
        Args:
            input_ids: [batch_size, seq_len] - concatenated query-document pairs
            labels: [batch_size] - relevance scores or binary labels
        """
        # Get embeddings and encode
        embeddings = self.forward_embeddings(input_ids)
        hidden_states = self.forward_encoder(embeddings, attention_mask)
        
        # Pool
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        else:
            pooled_output = hidden_states.mean(dim=1)
        
        # Ranking score
        rank_score = self.rank_classifier(pooled_output).squeeze(-1)
        
        outputs = {
            'scores': rank_score,
            'pooled_output': pooled_output
        }
        
        # Compute loss if labels provided
        if labels is not None:
            if labels.dtype == torch.float:
                # Regression loss for continuous scores
                loss = F.mse_loss(rank_score, labels)
            else:
                # Binary classification loss
                loss = F.binary_cross_entropy_with_logits(rank_score, labels.float())
            
            outputs['loss'] = loss
        
        return outputs


def create_task_model(task: str, config: TopoformerConfig, **kwargs) -> nn.Module:
    """Factory function to create task-specific models"""
    
    if task == 'classification':
        num_labels = kwargs.get('num_labels', 2)
        label_hierarchy = kwargs.get('label_hierarchy', None)
        return TopoformerForMultiLabelClassification(config, num_labels, label_hierarchy)
    
    elif task == 'qa':
        return TopoformerForQuestionAnswering(config)
    
    elif task == 'retrieval':
        projection_dim = kwargs.get('projection_dim', 256)
        return TopoformerForRetrieval(config, projection_dim)
    
    elif task == 'ranking':
        return TopoformerForSequenceRanking(config)
    
    else:
        raise ValueError(f"Unknown task: {task}")


def test_task_models():
    """Test different task models"""
    config = TopoformerConfig(
        vocab_size=30000,
        embed_dim=256,
        num_layers=3,
        num_heads=4,
        k_neighbors=16
    )
    
    batch_size = 4
    seq_len = 128
    
    # Test classification
    print("Testing Classification Model:")
    clf_model = create_task_model('classification', config, num_labels=14)
    input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    labels = torch.randint(0, 2, (batch_size, 14)).float()
    
    outputs = clf_model(input_ids, labels=labels)
    print(f"Classification loss: {outputs['loss'].item():.4f}")
    
    # Test QA
    print("\nTesting QA Model:")
    qa_model = create_task_model('qa', config)
    start_pos = torch.randint(0, seq_len, (batch_size,))
    end_pos = torch.randint(0, seq_len, (batch_size,))
    
    outputs = qa_model(input_ids, start_positions=start_pos, end_positions=end_pos)
    print(f"QA loss: {outputs['loss'].item():.4f}")
    
    # Test Retrieval
    print("\nTesting Retrieval Model:")
    retrieval_model = create_task_model('retrieval', config)
    query_ids = torch.randint(0, 30000, (batch_size, seq_len // 2))
    doc_ids = torch.randint(0, 30000, (batch_size, seq_len))
    
    outputs = retrieval_model(query_ids, doc_ids)
    print(f"Similarity shape: {outputs['similarities'].shape}")
    
    print("\nAll task models tested successfully!")


if __name__ == "__main__":
    test_task_models()