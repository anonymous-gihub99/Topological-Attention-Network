"""
Streamlined Topoformer Implementation
Optimized for notebook execution with synthetic bug dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TopoformerConfig:
    """Configuration for Topoformer model"""
    vocab_size: int = 50000
    embed_dim: int = 768
    num_layers: int = 6
    num_heads: int = 12
    max_seq_len: int = 512
    dropout: float = 0.1
    k_neighbors: int = 32
    use_topology: bool = True
    num_labels: int = 5  # For component classification
    
    # Hierarchical label structure for bugs
    num_components: int = 5
    num_subcomponents: int = 20
    num_bug_types: int = 5
    num_severities: int = 4


class SimplifiedTopologyExtractor(nn.Module):
    """Simplified topology extraction for faster training"""
    
    def __init__(self, embed_dim: int, k_neighbors: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.k_neighbors = k_neighbors
        
        # Learnable parameters for topology
        self.topology_proj = nn.Linear(embed_dim, embed_dim // 2)
        self.topology_encoder = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Extract topological features from embeddings
        
        Args:
            embeddings: [batch_size, seq_len, embed_dim]
            
        Returns:
            topo_features: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = embeddings.shape
        
        # Project embeddings for topology computation
        topo_proj = self.topology_proj(embeddings)  # [batch, seq_len, embed_dim//2]
        
        # Compute pairwise distances
        distances = torch.cdist(topo_proj, topo_proj, p=2)  # [batch, seq_len, seq_len]
        
        # Get k-nearest neighbors
        _, indices = torch.topk(distances, self.k_neighbors, dim=-1, largest=False)
        
        # Aggregate neighbor features
        neighbor_features = torch.gather(
            embeddings.unsqueeze(2).expand(-1, -1, seq_len, -1),
            2,
            indices.unsqueeze(-1).expand(-1, -1, -1, self.embed_dim)
        )  # [batch, seq_len, k_neighbors, embed_dim]
        
        # Mean pooling over neighbors
        pooled_features = neighbor_features.mean(dim=2)  # [batch, seq_len, embed_dim]
        
        # Encode topological features
        topo_features = self.topology_encoder(pooled_features[:, :, :self.embed_dim//2])
        
        return topo_features


class TopologicalAttention(nn.Module):
    """Attention mechanism with topological features"""
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Standard attention components
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Topology integration
        self.topo_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, topo_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Standard multi-head attention
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e4)
        
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        attn_output = torch.matmul(attn_probs, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)
        
        # Gate with topological features
        combined = torch.cat([attn_output, topo_features], dim=-1)
        gate = self.topo_gate(combined)
        
        output = self.out_proj(attn_output * gate + topo_features * (1 - gate))
        
        return output


class TopoformerLayer(nn.Module):
    """Single Topoformer layer"""
    
    def __init__(self, config: TopoformerConfig):
        super().__init__()
        self.config = config
        
        # Topology extractor
        self.topology_extractor = SimplifiedTopologyExtractor(
            config.embed_dim, 
            config.k_neighbors
        )
        
        # Topological attention
        self.attention = TopologicalAttention(
            config.embed_dim,
            config.num_heads
        )
        
        # Standard transformer components
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim * 4, config.embed_dim),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Extract topological features
        if self.config.use_topology:
            topo_features = self.topology_extractor(x)
        else:
            topo_features = torch.zeros_like(x)
        
        # Attention with topology
        attn_output = self.attention(x, topo_features, mask)
        x = self.norm1(x + attn_output)
        
        # FFN
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x


class TopoformerForBugClassification(nn.Module):
    """Topoformer model for hierarchical bug classification"""
    
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
        
        # Hierarchical classification heads
        self.component_classifier = nn.Linear(config.embed_dim, config.num_components)
        self.subcomponent_classifier = nn.Linear(
            config.embed_dim + config.num_components, 
            config.num_subcomponents
        )
        self.bugtype_classifier = nn.Linear(
            config.embed_dim + config.num_subcomponents,
            config.num_bug_types
        )
        self.severity_classifier = nn.Linear(config.embed_dim, config.num_severities)
        
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
                labels: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with hierarchical classification
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: Dictionary with 'component', 'sub_component', 'bug_type', 'severity'
            
        Returns:
            Dictionary with logits and optionally loss
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
        
        # Hierarchical classification
        component_logits = self.component_classifier(pooled_output)
        
        # Use component predictions to inform subcomponent classification
        component_probs = F.softmax(component_logits, dim=-1)
        subcomponent_input = torch.cat([pooled_output, component_probs], dim=-1)
        subcomponent_logits = self.subcomponent_classifier(subcomponent_input)
        
        # Use subcomponent predictions for bug type
        subcomponent_probs = F.softmax(subcomponent_logits, dim=-1)
        bugtype_input = torch.cat([pooled_output, subcomponent_probs], dim=-1)
        bugtype_logits = self.bugtype_classifier(bugtype_input)
        
        # Severity is independent
        severity_logits = self.severity_classifier(pooled_output)
        
        outputs = {
            'component_logits': component_logits,
            'subcomponent_logits': subcomponent_logits,
            'bugtype_logits': bugtype_logits,
            'severity_logits': severity_logits,
            'pooled_output': pooled_output
        }
        
        # Compute loss if labels provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            
            component_loss = loss_fn(component_logits, labels['component'])
            subcomponent_loss = loss_fn(subcomponent_logits, labels['sub_component'])
            bugtype_loss = loss_fn(bugtype_logits, labels['bug_type'])
            severity_loss = loss_fn(severity_logits, labels['severity'])
            
            # Weighted combination of losses
            total_loss = (
                0.3 * component_loss + 
                0.3 * subcomponent_loss + 
                0.2 * bugtype_loss + 
                0.2 * severity_loss
            )
            
            outputs['loss'] = total_loss
            outputs['losses'] = {
                'component': component_loss,
                'subcomponent': subcomponent_loss,
                'bugtype': bugtype_loss,
                'severity': severity_loss
            }
        
        return outputs


def create_model(config: Optional[TopoformerConfig] = None) -> TopoformerForBugClassification:
    """Create Topoformer model with default config"""
    if config is None:
        config = TopoformerConfig()
    
    model = TopoformerForBugClassification(config)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Created Topoformer model:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Configuration: {config}")
    
    return model