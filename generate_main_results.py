"""
Generate Main Results for TAN Paper - Tables 1 & 2
Produces classification results on ArXiv/LEDGAR and long document performance on BookSum
Author: TAN Research Team
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
from transformers import AutoTokenizer, BartTokenizer
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import gc
import psutil
import GPUtil

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import TAN components
from streamlined_topoformer import TopoformerConfig, TopoformerLayer
from train_tan_arxiv_dataparallel import TopoformerForSingleLabelClassification
from tan_summarization_dataparallel import TopoformerForSummarization
from baseline_comparison_fixed import (
    BERTTopological, TopoLM, HAT, S4Model, 
    HTransformer1D, GraphFormers, StructFormer, HTNN,
    SimpleDataset, MemoryTracker
)

# Configuration
DATASET_CONFIGS = {
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
    },
    'booksum': {
        'vocab_size': 50265,
        'embed_dim': 768,
        'num_layers': 6,
        'num_heads': 12,
        'max_seq_length': 16384,
        'num_classes': 1
    }
}

@dataclass
class ModelMetrics:
    """Store comprehensive model metrics"""
    model_name: str
    dataset: str
    task: str
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_macro: float = 0.0
    f1_micro: float = 0.0
    rouge1: float = 0.0
    rouge2: float = 0.0
    rougeL: float = 0.0
    time_per_sample: float = 0.0
    memory_gpu_gb: float = 0.0
    memory_cpu_gb: float = 0.0
    num_parameters: int = 0
    inference_time_ms: float = 0.0
    
    def to_dict(self):
        return asdict(self)


class TANEvaluator:
    """Main evaluator class for TAN and baselines"""
    
    def __init__(self, model_dir: str = './', data_dir: str = './processed_data'):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.memory_tracker = MemoryTracker()
        
    def load_tan_model(self, dataset: str, task: str = 'classification') -> nn.Module:
        """Load trained TAN model"""
        if dataset == 'arxiv':
            model_path = self.model_dir / 'arxiv_models' / task / f'best_model_{task}.pt'
        elif dataset == 'ledgar':
            model_path = self.model_dir / 'ledgar_models' / task / f'best_model_{task}.pt'
        elif dataset == 'booksum':
            model_path = self.model_dir / 'booksum_models' / f'best_model_summarization.pt'
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return None
        
        # Load configuration
        config_dict = DATASET_CONFIGS[dataset]
        config = TopoformerConfig(
            vocab_size=config_dict['vocab_size'],
            embed_dim=config_dict['embed_dim'],
            num_layers=config_dict['num_layers'],
            num_heads=config_dict['num_heads'],
            max_seq_len=config_dict['max_seq_length'],
            k_neighbors=32,
            use_topology=True,
            num_labels=config_dict['num_classes'],
            dropout=0.1
        )
        
        # Create model
        if dataset == 'booksum':
            model = TopoformerForSummarization(config)
        else:
            model = TopoformerForSingleLabelClassification(config, config_dict['num_classes'])
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        logger.info(f"Loaded TAN model from {model_path}")
        return model
    
    def create_baseline_model(self, model_name: str, config: Dict) -> nn.Module:
        """Create baseline model"""
        model_classes = {
            'BERT-Topological': BERTTopological,
            'Longformer': self._create_longformer,
            'BigBird': self._create_bigbird,
            'HAT': HAT,
            'S4': S4Model,
            'H-Transformer-1D': HTransformer1D,
            'GraphFormers': GraphFormers,
            'StructFormer': StructFormer,
            'HTNN': HTNN
        }
        
        if model_name in model_classes:
            if model_name in ['Longformer', 'BigBird']:
                model = model_classes[model_name](config)
            else:
                model = model_classes[model_name](config)
            model = model.to(self.device)
            model.eval()
            return model
        else:
            logger.warning(f"Unknown model: {model_name}")
            return None
    
    def _create_longformer(self, config: Dict) -> nn.Module:
        """Create Longformer model"""
        from transformers import LongformerModel, LongformerConfig
        
        longformer_config = LongformerConfig(
            vocab_size=config['vocab_size'],
            hidden_size=config['embed_dim'],
            num_hidden_layers=config['num_layers'],
            num_attention_heads=config['num_heads'],
            max_position_embeddings=config['max_seq_length'],
            attention_window=[256] * config['num_layers']
        )
        
        class LongformerClassifier(nn.Module):
            def __init__(self, config_dict):
                super().__init__()
                self.longformer = LongformerModel(longformer_config)
                self.classifier = nn.Linear(config_dict['embed_dim'], config_dict['num_classes'])
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, input_ids, attention_mask=None, **kwargs):
                outputs = self.longformer(input_ids=input_ids, attention_mask=attention_mask)
                pooled = outputs.last_hidden_state[:, 0]
                pooled = self.dropout(pooled)
                logits = self.classifier(pooled)
                return {'logits': logits}
        
        return LongformerClassifier(config)
    
    def _create_bigbird(self, config: Dict) -> nn.Module:
        """Create BigBird model"""
        from transformers import BigBirdModel, BigBirdConfig
        
        bigbird_config = BigBirdConfig(
            vocab_size=config['vocab_size'],
            hidden_size=config['embed_dim'],
            num_hidden_layers=config['num_layers'],
            num_attention_heads=config['num_heads'],
            max_position_embeddings=config['max_seq_length'],
            attention_type="block_sparse"
        )
        
        class BigBirdClassifier(nn.Module):
            def __init__(self, config_dict):
                super().__init__()
                self.bigbird = BigBirdModel(bigbird_config)
                self.classifier = nn.Linear(config_dict['embed_dim'], config_dict['num_classes'])
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, input_ids, attention_mask=None, **kwargs):
                outputs = self.bigbird(input_ids=input_ids, attention_mask=attention_mask)
                pooled = outputs.last_hidden_state[:, 0]
                pooled = self.dropout(pooled)
                logits = self.classifier(pooled)
                return {'logits': logits}
        
        return BigBirdClassifier(config)
    
    def evaluate_classification(self, model: nn.Module, dataloader: DataLoader, 
                              model_name: str) -> Dict[str, float]:
        """Evaluate classification performance"""
        model.eval()
        all_preds = []
        all_labels = []
        total_time = 0
        num_samples = 0
        
        self.memory_tracker.reset()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {model_name}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.memory_tracker.update()
                
                # Time inference
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.perf_counter()
                
                # Forward pass
                try:
                    outputs = model(input_ids, attention_mask)
                    
                    if isinstance(outputs, dict) and 'logits' in outputs:
                        logits = outputs['logits']
                    elif hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    else:
                        logits = outputs
                    
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    end_time = time.perf_counter()
                    
                    total_time += (end_time - start_time)
                    num_samples += input_ids.size(0)
                    
                    preds = torch.argmax(logits, dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                except Exception as e:
                    logger.error(f"Error in {model_name}: {e}")
                    continue
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        f1_micro = f1_score(all_labels, all_preds, average='micro')
        
        memory_stats = self.memory_tracker.get_peak_memory()
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'time_per_sample': total_time / num_samples * 1000,  # Convert to ms
            'memory_gpu_gb': memory_stats['gpu_peak_gb'],
            'memory_cpu_gb': memory_stats['cpu_peak_gb'],
            'predictions': all_preds,
            'labels': all_labels
        }
    
    def evaluate_summarization(self, model: nn.Module, dataloader: DataLoader,
                              model_name: str, max_length: int) -> Dict[str, float]:
        """Evaluate summarization performance"""
        from rouge_score import rouge_scorer
        
        model.eval()
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        all_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        total_time = 0
        num_samples = 0
        
        self.memory_tracker.reset()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {model_name} at {max_length} tokens"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                references = batch['reference']
                
                self.memory_tracker.update()
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.perf_counter()
                
                try:
                    # Generate summary
                    if hasattr(model, 'generate'):
                        generated_ids = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_length=150,
                            num_beams=4,
                            early_stopping=True
                        )
                    else:
                        # Simple greedy decoding for models without generate
                        outputs = model(input_ids, attention_mask)
                        generated_ids = torch.argmax(outputs['logits'], dim=-1)
                    
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    end_time = time.perf_counter()
                    
                    total_time += (end_time - start_time)
                    num_samples += input_ids.size(0)
                    
                    # Decode and score
                    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
                    for i in range(len(references)):
                        generated_text = tokenizer.decode(generated_ids[i], skip_special_tokens=True)
                        scores = scorer.score(references[i], generated_text)
                        
                        for metric in all_scores:
                            all_scores[metric].append(scores[metric].fmeasure)
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        logger.warning(f"{model_name} OOM at {max_length} tokens")
                        return {'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'OOM': True}
                    else:
                        raise e
        
        memory_stats = self.memory_tracker.get_peak_memory()
        
        return {
            'rouge1': np.mean(all_scores['rouge1']),
            'rouge2': np.mean(all_scores['rouge2']),
            'rougeL': np.mean(all_scores['rougeL']),
            'time_per_sample': total_time / num_samples,
            'memory_gpu_gb': memory_stats['gpu_peak_gb'],
            'memory_cpu_gb': memory_stats['cpu_peak_gb']
        }
    
    def generate_table1(self, output_dir: Path) -> pd.DataFrame:
        """Generate Table 1: Classification Performance on ArXiv and LEDGAR"""
        logger.info("Generating Table 1: Classification Performance")
        
        models = ['TAN', 'BERT-Topological', 'Longformer', 'BigBird', 
                  'HAT', 'S4', 'H-Transformer-1D', 'GraphFormers', 'StructFormer']
        
        results = []
        
        for dataset in ['arxiv', 'ledgar']:
            logger.info(f"\nEvaluating {dataset.upper()} dataset")
            
            # Load tokenizer and data
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            test_dataset = SimpleDataset(self.data_dir, 'test', tokenizer, dataset, 'classification')
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            config = DATASET_CONFIGS[dataset]
            
            for model_name in models:
                logger.info(f"  Evaluating {model_name}...")
                
                # Load or create model
                if model_name == 'TAN':
                    model = self.load_tan_model(dataset, 'classification')
                else:
                    model = self.create_baseline_model(model_name, config)
                
                if model is None:
                    continue
                
                # Count parameters
                num_params = sum(p.numel() for p in model.parameters()) / 1e6  # Convert to millions
                
                # Evaluate
                metrics = self.evaluate_classification(model, test_loader, model_name)
                
                result = {
                    'Model': model_name,
                    f'{dataset.upper()} Acc': metrics['accuracy'],
                    f'{dataset.upper()} F1': metrics['f1_macro'],
                    'Params (M)': num_params,
                    'Time (ms)': metrics['time_per_sample']
                }
                results.append(result)
                
                # Clean up
                del model
                torch.cuda.empty_cache()
                gc.collect()
        
        # Create DataFrame and format
        df = pd.DataFrame(results)
        df = df.pivot_table(index='Model', values=[col for col in df.columns if col != 'Model'])
        
        # Save as CSV
        df.to_csv(output_dir / 'table1_classification_results.csv')
        
        # Generate LaTeX
        latex = self._generate_table1_latex(df)
        with open(output_dir / 'table1_classification.tex', 'w') as f:
            f.write(latex)
        
        logger.info(f"Table 1 saved to {output_dir}")
        return df
    
    def generate_table2(self, output_dir: Path) -> pd.DataFrame:
        """Generate Table 2: BookSum Summarization Results"""
        logger.info("Generating Table 2: Long Document Performance")
        
        models = ['TAN', 'Longformer', 'BigBird', 'HAT', 'S4']
        doc_lengths = [2048, 4096, 8192, 16384]
        
        results = []
        
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        
        for max_length in doc_lengths:
            logger.info(f"\nEvaluating at {max_length} tokens")
            
            # Update config
            config = DATASET_CONFIGS['booksum'].copy()
            config['max_seq_length'] = max_length
            
            # Create dataset with limited length
            test_dataset = SimpleDataset(self.data_dir, 'test', tokenizer, 'booksum', 'summarization')
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
            
            for model_name in models:
                logger.info(f"  Evaluating {model_name}...")
                
                # Load or create model
                if model_name == 'TAN':
                    model = self.load_tan_model('booksum')
                else:
                    model = self.create_baseline_model(model_name, config)
                
                if model is None:
                    continue
                
                # Evaluate
                metrics = self.evaluate_summarization(model, test_loader, model_name, max_length)
                
                if 'OOM' in metrics:
                    result = {
                        'Model': model_name,
                        'Length': max_length,
                        'ROUGE-L': 'OOM',
                        'Memory (GB)': 'OOM',
                        'Time (s)': 'OOM'
                    }
                else:
                    result = {
                        'Model': model_name,
                        'Length': max_length,
                        'ROUGE-L': metrics['rougeL'],
                        'Memory (GB)': metrics['memory_gpu_gb'],
                        'Time (s)': metrics['time_per_sample']
                    }
                results.append(result)
                
                # Clean up
                del model
                torch.cuda.empty_cache()
                gc.collect()
        
        # Create DataFrame
        df = pd.DataFrame(results)
        df_pivot = df.pivot(index='Model', columns='Length', values=['ROUGE-L', 'Memory (GB)', 'Time (s)'])
        
        # Save as CSV
        df_pivot.to_csv(output_dir / 'table2_booksum_results.csv')
        
        # Generate LaTeX
        latex = self._generate_table2_latex(df_pivot)
        with open(output_dir / 'table2_booksum.tex', 'w') as f:
            f.write(latex)
        
        logger.info(f"Table 2 saved to {output_dir}")
        return df_pivot
    
    def _generate_table1_latex(self, df: pd.DataFrame) -> str:
        """Generate LaTeX for Table 1"""
        latex = r"""\begin{table}[htbp]
\centering
\caption{Classification Performance on ArXiv and LEDGAR Datasets}
\label{tab:main_results}
\begin{tabular}{lcccccc}
\toprule
Model & ArXiv Acc & ArXiv F1 & LEDGAR Acc & LEDGAR F1 & Params (M) & Time (ms) \\
\midrule
"""
        for idx, row in df.iterrows():
            model_name = idx
            if model_name == 'TAN':
                # Bold TAN results
                latex += f"\\textbf{{{model_name}}} & "
                latex += f"\\textbf{{{row['ArXiv Acc']:.3f}}} & "
                latex += f"\\textbf{{{row['ArXiv F1']:.3f}}} & "
                latex += f"\\textbf{{{row['LEDGAR Acc']:.3f}}} & "
                latex += f"\\textbf{{{row['LEDGAR F1']:.3f}}} & "
                latex += f"\\textbf{{{row['Params (M)']:.0f}}} & "
                latex += f"\\textbf{{{row['Time (ms)']:.1f}}} \\\\\n"
            else:
                latex += f"{model_name} & "
                latex += f"{row['ArXiv Acc']:.3f} & "
                latex += f"{row['ArXiv F1']:.3f} & "
                latex += f"{row['LEDGAR Acc']:.3f} & "
                latex += f"{row['LEDGAR F1']:.3f} & "
                latex += f"{row['Params (M)']:.0f} & "
                latex += f"{row['Time (ms)']:.1f} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table}"""
        return latex
    
    def _generate_table2_latex(self, df: pd.DataFrame) -> str:
        """Generate LaTeX for Table 2"""
        latex = r"""\begin{table*}[htbp]
\centering
\caption{Performance on Long Documents from BookSum Dataset}
\label{tab:long_docs}
\begin{tabular}{l|ccc|ccc|ccc|ccc}
\toprule
\multirow{2}{*}{Model} & \multicolumn{3}{c|}{2K tokens} & \multicolumn{3}{c|}{4K tokens} & \multicolumn{3}{c|}{8K tokens} & \multicolumn{3}{c}{16K tokens} \\
\cmidrule{2-13}
& R-L & Mem & Time & R-L & Mem & Time & R-L & Mem & Time & R-L & Mem & Time \\
\midrule
"""
        
        for idx, row in df.iterrows():
            model_name = idx
            if model_name == 'TAN':
                latex += f"\\textbf{{{model_name}}} & "
            else:
                latex += f"{model_name} & "
            
            for length in [2048, 4096, 8192, 16384]:
                rouge = row[('ROUGE-L', length)]
                mem = row[('Memory (GB)', length)]
                time = row[('Time (s)', length)]
                
                if rouge == 'OOM':
                    latex += "OOM & OOM & OOM"
                else:
                    if model_name == 'TAN':
                        latex += f"\\textbf{{{rouge:.3f}}} & \\textbf{{{mem:.1f}}} & \\textbf{{{time:.2f}}}"
                    else:
                        latex += f"{rouge:.3f} & {mem:.1f} & {time:.2f}"
                
                if length < 16384:
                    latex += " & "
            
            latex += " \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table*}"""
        return latex


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate main results for TAN paper')
    parser.add_argument('--model-dir', type=str, default='./', 
                       help='Directory containing trained models')
    parser.add_argument('--data-dir', type=str, default='./processed_data',
                       help='Directory containing processed datasets')
    parser.add_argument('--output-dir', type=str, default='./paper_results',
                       help='Directory to save results')
    parser.add_argument('--tables', nargs='+', default=['table1', 'table2'],
                       choices=['table1', 'table2'],
                       help='Which tables to generate')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize evaluator
    evaluator = TANEvaluator(model_dir=args.model_dir, data_dir=args.data_dir)
    
    # Generate requested tables
    if 'table1' in args.tables:
        logger.info("="*60)
        logger.info("Generating Table 1: Classification Results")
        logger.info("="*60)
        df_table1 = evaluator.generate_table1(output_dir)
        print("\nTable 1 Results:")
        print(df_table1)
    
    if 'table2' in args.tables:
        logger.info("="*60)
        logger.info("Generating Table 2: Long Document Results")
        logger.info("="*60)
        df_table2 = evaluator.generate_table2(output_dir)
        print("\nTable 2 Results:")
        print(df_table2)
    
    logger.info(f"\nAll results saved to {output_dir}")
    logger.info("Generation complete!")


if __name__ == "__main__":
    main()