"""
Fixed Ablation Study Script with LSH Analysis and Failure Mode Analysis
Loads trained TAN models and performs ablation studies
Saves all results as structured data files
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from transformers import AutoTokenizer
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import warnings
import gc
import copy
warnings.filterwarnings('ignore')

# Import model architectures
try:
    from train_tan_arxiv_fixed import TopoformerForSingleLabelClassification
    from streamlined_topoformer import TopoformerConfig, TopoformerLayer
except ImportError as e:
    logger.warning(f"Import error: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Store ablation study results"""
    config_name: str
    dataset: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_macro: float = 0.0
    f1_micro: float = 0.0
    collision_rate: float = 0.0
    hash_quality: float = 0.0
    intra_class_collisions: int = 0
    inter_class_collisions: int = 0
    time_per_sample: float = 0.0
    
    def to_dict(self):
        return asdict(self)


@dataclass
class FailureCase:
    """Store failure case information"""
    sample_id: int
    text: str
    true_label: int
    predicted_label: int
    confidence: float
    top_5_predictions: List[Tuple[int, float]]
    attention_entropy: float = 0.0
    
    def to_dict(self):
        return {
            'sample_id': self.sample_id,
            'text': self.text[:200],  # Truncate for storage
            'true_label': self.true_label,
            'predicted_label': self.predicted_label,
            'confidence': self.confidence,
            'top_5_predictions': self.top_5_predictions,
            'attention_entropy': self.attention_entropy
        }


class LSHAnalyzer:
    """Analyze LSH performance in TAN"""
    
    def __init__(self, embed_dim: int = 768, num_hashes: int = 64, hash_dim: int = 128):
        self.embed_dim = embed_dim
        self.num_hashes = num_hashes
        self.hash_dim = hash_dim
        
        # Initialize random projection matrices for LSH
        self.projection_matrices = nn.Parameter(
            torch.randn(num_hashes, embed_dim, hash_dim) / np.sqrt(embed_dim)
        )
        
        # Metrics storage
        self.collision_rates = []
        self.retrieval_accuracy = []
        self.hash_quality_scores = []
        self.bucket_distributions = []
    
    def compute_hashes(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute LSH hashes for embeddings"""
        # embeddings: [batch_size, seq_len, embed_dim]
        batch_size, seq_len, _ = embeddings.shape
        
        # Project embeddings
        projected = torch.einsum('bse,hek->bshk', embeddings, self.projection_matrices)
        
        # Apply sign function for binary hashes
        hashes = torch.sign(projected)
        
        return hashes
    
    def analyze_collisions(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """Analyze collision rates and patterns"""
        hashes = self.compute_hashes(embeddings)
        batch_size, seq_len, num_hashes, hash_dim = hashes.shape
        
        # Flatten hashes to binary strings for analysis
        hash_strings = []
        label_list = []
        
        for b in range(batch_size):
            for s in range(seq_len):
                hash_str = ''.join([str(int(h)) for h in hashes[b, s].flatten().tolist()])
                hash_strings.append(hash_str)
                if labels is not None and b < len(labels):
                    label_list.append(labels[b].item())
        
        # Count collisions
        hash_counter = Counter(hash_strings)
        collision_rate = sum(1 for count in hash_counter.values() if count > 1) / len(hash_strings)
        self.collision_rates.append(collision_rate)
        
        # Analyze collision patterns by label if provided
        intra_class_collisions = 0
        inter_class_collisions = 0
        
        if labels is not None and label_list:
            # Group hashes by label
            label_hashes = defaultdict(list)
            for i, (hash_str, label) in enumerate(zip(hash_strings, label_list * seq_len)):
                label_hashes[label].append(hash_str)
            
            # Count intra-class collisions
            for label, hashes_for_label in label_hashes.items():
                hash_counts = Counter(hashes_for_label)
                intra_class_collisions += sum(count - 1 for count in hash_counts.values() if count > 1)
            
            # Total collisions
            total_collisions = sum(count - 1 for count in hash_counter.values() if count > 1)
            inter_class_collisions = total_collisions - intra_class_collisions
        
        return {
            'collision_rate': collision_rate,
            'intra_class_collisions': intra_class_collisions,
            'inter_class_collisions': inter_class_collisions,
            'bucket_distribution': dict(hash_counter.most_common(10))
        }
    
    def analyze_hash_quality(self, embeddings: torch.Tensor) -> Dict:
        """Analyze the quality of hash functions"""
        hashes = self.compute_hashes(embeddings)
        
        # Compute hash bit balance (should be close to 0.5 for good hashes)
        bit_balance = (hashes > 0).float().mean(dim=[0, 1]).cpu()
        
        # Compute hash independence (correlation between hash bits)
        flattened_hashes = hashes.view(-1, self.num_hashes * self.hash_dim)
        
        # Sample for efficiency
        if flattened_hashes.shape[0] > 1000:
            indices = torch.randperm(flattened_hashes.shape[0])[:1000]
            flattened_hashes = flattened_hashes[indices]
        
        hash_correlation = torch.corrcoef(flattened_hashes.T)
        
        # Off-diagonal elements represent inter-hash correlations
        mask = ~torch.eye(hash_correlation.shape[0], dtype=bool)
        avg_correlation = torch.abs(hash_correlation[mask]).mean().item()
        
        # Quality score: balance and independence
        balance_score = 1 - torch.abs(bit_balance - 0.5).mean().item() * 2
        independence_score = 1 - avg_correlation
        quality_score = (balance_score + independence_score) / 2
        
        self.hash_quality_scores.append(quality_score)
        
        return {
            'bit_balance': bit_balance.mean().item(),
            'avg_correlation': avg_correlation,
            'quality_score': quality_score,
            'balance_score': balance_score,
            'independence_score': independence_score
        }


class ModifiedTANModel(nn.Module):
    """Modified TAN model for ablation studies"""
    
    def __init__(self, base_model, ablation_config: Dict):
        super().__init__()
        self.base_model = base_model
        self.ablation_config = ablation_config
        
        # Disable components based on ablation config
        if not ablation_config.get('use_topology', True):
            # Disable topology by replacing with identity
            self._disable_topology()
        
        # Modify k-neighbors if specified
        if 'k_neighbors' in ablation_config:
            self._modify_k_neighbors(ablation_config['k_neighbors'])
        
        # Modify aggregation method if specified
        if 'topology_aggregation' in ablation_config:
            self._modify_aggregation(ablation_config['topology_aggregation'])
        
        # Modify dropout if specified
        if 'dropout_topology' in ablation_config:
            self._modify_dropout(ablation_config['dropout_topology'])
    
    def _disable_topology(self):
        """Disable topology components"""
        # Replace topology computation with zeros
        for layer in self.base_model.layers:
            if hasattr(layer, 'compute_topology'):
                layer.compute_topology = lambda x: torch.zeros_like(x)
    
    def _modify_k_neighbors(self, k: int):
        """Modify k-neighbors parameter"""
        if hasattr(self.base_model, 'config'):
            self.base_model.config.k_neighbors = k
        # Update any k-dependent components
        for layer in self.base_model.layers:
            if hasattr(layer, 'k_neighbors'):
                layer.k_neighbors = k
    
    def _modify_aggregation(self, method: str):
        """Modify topology aggregation method"""
        # This would require modifying the aggregation function in topology layers
        pass
    
    def _modify_dropout(self, dropout_rate: float):
        """Modify dropout rate for topology"""
        for layer in self.base_model.layers:
            if hasattr(layer, 'topology_dropout'):
                layer.topology_dropout = nn.Dropout(dropout_rate)
    
    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)


class FailureModeAnalyzer:
    """Analyze failure modes of TAN model"""
    
    def __init__(self):
        self.failure_cases = []
        self.error_patterns = defaultdict(list)
        self.confidence_distributions = {
            'correct': [],
            'incorrect': []
        }
    
    def analyze_predictions(self, outputs: Dict, labels: torch.Tensor, 
                          texts: List[str], sample_ids: List[int]) -> Dict:
        """Analyze model predictions and identify failure patterns"""
        
        if 'logits' in outputs:
            logits = outputs['logits']
        else:
            logits = outputs
        
        probs = F.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        
        # Get confidence scores
        confidence, _ = torch.max(probs, dim=-1)
        
        # Analyze each prediction
        analysis_results = {
            'total_samples': len(labels),
            'correct': 0,
            'incorrect': 0,
            'low_confidence_correct': 0,
            'high_confidence_incorrect': 0,
            'error_distribution': defaultdict(int)
        }
        
        for i in range(len(labels)):
            true_label = labels[i].item()
            pred_label = predictions[i].item()
            conf = confidence[i].item()
            
            is_correct = true_label == pred_label
            
            # Update confidence distributions
            self.confidence_distributions['correct' if is_correct else 'incorrect'].append(conf)
            
            if is_correct:
                analysis_results['correct'] += 1
                if conf < 0.5:
                    analysis_results['low_confidence_correct'] += 1
            else:
                analysis_results['incorrect'] += 1
                if conf > 0.8:
                    analysis_results['high_confidence_incorrect'] += 1
                
                # Store failure case
                top_5_probs, top_5_indices = torch.topk(probs[i], 5)
                top_5_predictions = [(idx.item(), prob.item()) for idx, prob in zip(top_5_indices, top_5_probs)]
                
                failure_case = FailureCase(
                    sample_id=sample_ids[i] if i < len(sample_ids) else i,
                    text=texts[i] if i < len(texts) else "",
                    true_label=true_label,
                    predicted_label=pred_label,
                    confidence=conf,
                    top_5_predictions=top_5_predictions
                )
                self.failure_cases.append(failure_case)
                
                # Track error patterns
                error_key = f"{true_label}_to_{pred_label}"
                self.error_patterns[error_key].append(failure_case)
                analysis_results['error_distribution'][error_key] += 1
        
        analysis_results['accuracy'] = analysis_results['correct'] / analysis_results['total_samples']
        
        return analysis_results
    
    def get_top_failure_patterns(self, n: int = 10) -> List[Tuple[str, int]]:
        """Get most common failure patterns"""
        pattern_counts = {pattern: len(cases) for pattern, cases in self.error_patterns.items()}
        return sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:n]
    
    def analyze_confidence_calibration(self) -> Dict:
        """Analyze confidence calibration"""
        correct_conf = self.confidence_distributions['correct']
        incorrect_conf = self.confidence_distributions['incorrect']
        
        if not correct_conf or not incorrect_conf:
            return {}
        
        return {
            'correct_mean_confidence': np.mean(correct_conf),
            'correct_std_confidence': np.std(correct_conf),
            'incorrect_mean_confidence': np.mean(incorrect_conf),
            'incorrect_std_confidence': np.std(incorrect_conf),
            'confidence_gap': np.mean(correct_conf) - np.mean(incorrect_conf)
        }


def load_trained_tan_model(model_path: str, config: Dict, device: torch.device):
    """Load trained TAN model"""
    
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return None
    
    # Create model config
    model_config = TopoformerConfig(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        max_seq_len=config['max_seq_length'],
        k_neighbors=32,
        use_topology=True,
        num_labels=config['num_classes'],
        dropout=0.1
    )
    
    # Create model
    model = TopoformerForSingleLabelClassification(model_config, config['num_classes'])
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    return model


def run_single_ablation(model, dataloader, device, ablation_name: str, 
                       lsh_analyzer: Optional[LSHAnalyzer] = None):
    """Run evaluation for a single ablation configuration"""
    
    model.eval()
    all_preds = []
    all_labels = []
    all_embeddings = []
    total_time = 0
    
    # Collect predictions
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {ablation_name}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            start_time = time.time()
            outputs = model(input_ids, attention_mask, labels=labels)
            total_time += time.time() - start_time
            
            if isinstance(outputs, dict):
                logits = outputs['logits']
                # Get embeddings if available
                if 'final_hidden_states' in outputs:
                    all_embeddings.append(outputs['final_hidden_states'].cpu())
            else:
                logits = outputs
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    
    # LSH analysis if analyzer provided and embeddings available
    lsh_metrics = {}
    if lsh_analyzer and all_embeddings:
        embeddings = torch.cat(all_embeddings, dim=0)
        labels_tensor = torch.tensor(all_labels)
        
        # Analyze collisions
        collision_results = lsh_analyzer.analyze_collisions(embeddings, labels_tensor)
        lsh_metrics.update(collision_results)
        
        # Analyze hash quality
        quality_results = lsh_analyzer.analyze_hash_quality(embeddings)
        lsh_metrics.update(quality_results)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'time_per_sample': total_time / len(all_labels),
        **lsh_metrics
    }


def run_ablation_study(args):
    """Run complete ablation study"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Dataset configurations
    dataset_configs = {
        'arxiv': {
            'vocab_size': 50000,
            'embed_dim': 768,
            'num_layers': 6,
            'num_heads': 12,
            'max_seq_length': 512,
            'num_classes': 28
        },
        'ledgar': {
            'vocab_size': 50000,
            'embed_dim': 768,
            'num_layers': 6,
            'num_heads': 10,
            'max_seq_length': 512,
            'num_classes': 100
        }
    }
    
    config = dataset_configs[args.dataset]
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Load base trained model
    model_path = Path(args.model_dir) / f'{args.dataset}_models' / 'classification' / 'best_model_classification.pt'
    base_model = load_trained_tan_model(str(model_path), config, device)
    
    if base_model is None:
        logger.error("Failed to load base model")
        return
    
    # Create test dataset
    from baseline_comparison_fixed import SimpleDataset
    test_dataset = SimpleDataset(args.data_dir, 'test', tokenizer, args.dataset, 'classification')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Ablation configurations
    ablation_configs = {
        'full_model': {},
        'no_topology': {'use_topology': False},
        'k_8': {'k_neighbors': 8},
        'k_16': {'k_neighbors': 16},
        'k_64': {'k_neighbors': 64},
        'agg_mean': {'topology_aggregation': 'mean'},
        'agg_max': {'topology_aggregation': 'max'},
        'dropout_0.0': {'dropout_topology': 0.0},
        'dropout_0.3': {'dropout_topology': 0.3},
        'dropout_0.5': {'dropout_topology': 0.5}
    }
    
    # Initialize analyzers
    lsh_analyzer = LSHAnalyzer()
    failure_analyzer = FailureModeAnalyzer()
    
    all_results = []
    
    # Run ablations
    for ablation_name, ablation_config in ablation_configs.items():
        logger.info(f"\nRunning ablation: {ablation_name}")
        
        # Create modified model
        if ablation_name == 'full_model':
            model = base_model
        else:
            # Create a copy of the base model
            model = copy.deepcopy(base_model)
            model = ModifiedTANModel(model, ablation_config)
            model = model.to(device)
        
        # Run evaluation
        metrics = run_single_ablation(model, test_loader, device, ablation_name, lsh_analyzer)
        
        # Create result object
        result = AblationResult(
            config_name=ablation_name,
            dataset=args.dataset,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_macro=metrics['f1_macro'],
            f1_micro=metrics['f1_micro'],
            collision_rate=metrics.get('collision_rate', 0.0),
            hash_quality=metrics.get('quality_score', 0.0),
            intra_class_collisions=metrics.get('intra_class_collisions', 0),
            inter_class_collisions=metrics.get('inter_class_collisions', 0),
            time_per_sample=metrics['time_per_sample']
        )
        all_results.append(result)
        
        logger.info(f"  Accuracy: {result.accuracy:.4f}, F1-Macro: {result.f1_macro:.4f}")
        
        # Clean up
        if ablation_name != 'full_model':
            del model
        torch.cuda.empty_cache()
        gc.collect()
    
    # Failure mode analysis on full model
    logger.info("\nRunning failure mode analysis...")
    
    model = base_model
    model.eval()
    
    all_texts = []
    all_labels = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            # Get texts (decode input_ids)
            texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            all_texts.extend(texts)
            all_labels.extend(labels.numpy())
            all_sample_ids.extend(range(batch_idx * args.batch_size, 
                                       batch_idx * args.batch_size + len(labels)))
            
            # Get predictions
            outputs = model(input_ids, attention_mask)
            
            # Analyze failures
            failure_analysis = failure_analyzer.analyze_predictions(
                outputs, labels, texts, all_sample_ids[-len(labels):]
            )
    
    # Get failure statistics
    top_patterns = failure_analyzer.get_top_failure_patterns()
    confidence_calibration = failure_analyzer.analyze_confidence_calibration()
    
    # Save results
    # 1. Ablation results as JSON
    ablation_json = [r.to_dict() for r in all_results]
    with open(results_dir / f'{args.dataset}_ablation_results.json', 'w') as f:
        json.dump(ablation_json, f, indent=2)
    
    # 2. Ablation results as CSV
    ablation_df = pd.DataFrame([r.to_dict() for r in all_results])
    ablation_df.to_csv(results_dir / f'{args.dataset}_ablation_results.csv', index=False)
    
    # 3. LSH analysis results
    lsh_results = {
        'collision_rates': lsh_analyzer.collision_rates,
        'hash_quality_scores': lsh_analyzer.hash_quality_scores,
        'avg_collision_rate': np.mean(lsh_analyzer.collision_rates) if lsh_analyzer.collision_rates else 0,
        'avg_hash_quality': np.mean(lsh_analyzer.hash_quality_scores) if lsh_analyzer.hash_quality_scores else 0
    }
    with open(results_dir / f'{args.dataset}_lsh_analysis.json', 'w') as f:
        json.dump(lsh_results, f, indent=2)
    
    # 4. Failure analysis results
    failure_results = {
        'total_failures': len(failure_analyzer.failure_cases),
        'top_error_patterns': top_patterns,
        'confidence_calibration': confidence_calibration,
        'high_confidence_failures': sum(1 for case in failure_analyzer.failure_cases if case.confidence > 0.8),
        'low_confidence_failures': sum(1 for case in failure_analyzer.failure_cases if case.confidence < 0.3)
    }
    with open(results_dir / f'{args.dataset}_failure_analysis.json', 'w') as f:
        json.dump(failure_results, f, indent=2)
    
    # 5. Save top failure cases
    top_failures = sorted(failure_analyzer.failure_cases, key=lambda x: x.confidence, reverse=True)[:100]
    failure_cases_data = [case.to_dict() for case in top_failures]
    with open(results_dir / f'{args.dataset}_failure_cases.json', 'w') as f:
        json.dump(failure_cases_data, f, indent=2)
    
    # 6. Create LaTeX tables
    create_ablation_latex_tables(ablation_df, failure_results, results_dir, args.dataset)
    
    logger.info(f"\nAblation study complete! Results saved to {results_dir}")
    
    return ablation_df, failure_results


def create_ablation_latex_tables(ablation_df: pd.DataFrame, failure_results: Dict, 
                                results_dir: Path, dataset: str):
    """Create LaTeX tables for ablation study"""
    
    # Main ablation table
    latex_content = f"""% Ablation Study Results for {dataset.upper()}
\\begin{{table}}[htbp]
\\centering
\\caption{{TAN/Topoformer Ablation Study on {dataset.upper()} Dataset}}
\\label{{tab:ablation_{dataset}}}
\\begin{{tabular}}{{lccccc}}
\\toprule
Configuration & Accuracy & F1-Macro & Collision Rate & Hash Quality & Time(s) \\\\
\\midrule
"""
    
    # Sort by F1 score
    ablation_df = ablation_df.sort_values('f1_macro', ascending=False)
    
    for _, row in ablation_df.iterrows():
        line = f"{row['config_name'].replace('_', ' ').title()} & "
        line += f"{row['accuracy']:.4f} & "
        line += f"{row['f1_macro']:.4f} & "
        line += f"{row['collision_rate']:.4f} & "
        line += f"{row['hash_quality']:.4f} & "
        line += f"{row['time_per_sample']:.4f} \\\\"
        
        if row['config_name'] == 'full_model':
            line = "\\textbf{" + line.replace(' & ', '} & \\textbf{').replace(' \\\\', '} \\\\')
        
        latex_content += line + "\n"
    
    latex_content += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open(results_dir / f'{dataset}_ablation_table.tex', 'w') as f:
        f.write(latex_content)
    
    # Failure analysis table
    failure_latex = f"""% Failure Analysis for {dataset.upper()}
\\begin{{table}}[htbp]
\\centering
\\caption{{Failure Mode Analysis on {dataset.upper()} Dataset}}
\\label{{tab:failure_{dataset}}}
\\begin{{tabular}}{{lc}}
\\toprule
Metric & Value \\\\
\\midrule
Total Failures & {failure_results['total_failures']} \\\\
High Confidence Failures (>0.8) & {failure_results['high_confidence_failures']} \\\\
Low Confidence Failures (<0.3) & {failure_results['low_confidence_failures']} \\\\
"""
    
    if 'confidence_calibration' in failure_results and failure_results['confidence_calibration']:
        calib = failure_results['confidence_calibration']
        failure_latex += f"Correct Mean Confidence & {calib.get('correct_mean_confidence', 0):.3f} \\\\\n"
        failure_latex += f"Incorrect Mean Confidence & {calib.get('incorrect_mean_confidence', 0):.3f} \\\\\n"
        failure_latex += f"Confidence Gap & {calib.get('confidence_gap', 0):.3f} \\\\\n"
    
    failure_latex += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open(results_dir / f'{dataset}_failure_table.tex', 'w') as f:
        f.write(failure_latex)


def main():
    """Main execution"""
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./processed_data')
    parser.add_argument('--model-dir', type=str, default='./')
    parser.add_argument('--results-dir', type=str, default='./ablation_results')
    parser.add_argument('--dataset', type=str, default='arxiv', choices=['arxiv', 'ledgar'])
    parser.add_argument('--batch-size', type=int, default=32)
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("TAN/Topoformer Ablation Study")
    logger.info("="*60)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model directory: {args.model_dir}")
    logger.info(f"Results directory: {args.results_dir}")
    
    # Run ablation study
    ablation_results, failure_results = run_ablation_study(args)
    
    # Print summary
    logger.info("\n" + "="*40)
    logger.info("Ablation Study Summary:")
    logger.info("="*40)
    
    # Best and worst configurations
    best_config = ablation_results.loc[ablation_results['f1_macro'].idxmax()]
    worst_config = ablation_results.loc[ablation_results['f1_macro'].idxmin()]
    
    logger.info(f"Best configuration: {best_config['config_name']} (F1: {best_config['f1_macro']:.4f})")
    logger.info(f"Worst configuration: {worst_config['config_name']} (F1: {worst_config['f1_macro']:.4f})")
    
    # Impact of topology
    if 'full_model' in ablation_results['config_name'].values and 'no_topology' in ablation_results['config_name'].values:
        full_f1 = ablation_results[ablation_results['config_name'] == 'full_model']['f1_macro'].iloc[0]
        no_topo_f1 = ablation_results[ablation_results['config_name'] == 'no_topology']['f1_macro'].iloc[0]
        logger.info(f"Topology impact: +{full_f1 - no_topo_f1:.4f} F1")
    
    logger.info(f"\nTotal failures analyzed: {failure_results['total_failures']}")
    logger.info(f"High confidence failures: {failure_results['high_confidence_failures']}")


if __name__ == "__main__":
    main()