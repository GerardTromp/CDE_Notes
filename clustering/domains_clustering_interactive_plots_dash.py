#!/usr/bin/env python3
"""
Interactive CDE clustering analysis with Dash and Plotly.
Data point selection with lasso/box & export tooltip data to JSON/CSV or clipboard
This version comes with faceted t-SNE/UMAP plots for SAPBERT/MedCPT
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import HDBSCAN
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap
import warnings
import time
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Dash imports
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pyperclip
from datetime import datetime

def setup_logging():
    """Configure logging for console (WARNING+) and file (DEBUG+)."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    file_handler = logging.FileHandler('interactive_clustering.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

logger = setup_logging()

class InteractiveClusteringAnalyzer:
    """Interactive CDE clustering analyzer with Dash interface."""
    
    def __init__(self, hdbscan_min_cluster_size: int = 15, hdbscan_min_samples: int = 5):
        """Initialize analyzer with HDBSCAN parameters."""
        self.hdbscan_params = {
            'min_cluster_size': hdbscan_min_cluster_size,
            'min_samples': hdbscan_min_samples,
            'metric': 'euclidean'
        }
        self._setup_gpu_config()
        self.d3_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'
        ]
        self.marker_shapes = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'star']
        self.embedding_models = {}
        self.tokenizers = {}
        self.domain_mapping = None
        self.cde_data = None
        self.filtered_cdes = None
        self.all_metrics = {}
        self.analysis_results = {}
        self.app = None
        logger.info(f"Initialized with HDBSCAN: min_cluster_size={hdbscan_min_cluster_size}, min_samples={hdbscan_min_samples}")
    
    def _setup_gpu_config(self) -> None:
        """Configure GPU for PyTorch models."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"Found {gpu_count} GPU(s)")
                self.device = torch.device("cuda:0")
                self.secondary_device = torch.device("cuda:1" if gpu_count >= 2 else "cuda:0")
                self.use_multi_gpu = gpu_count >= 2
                logger.info(f"{'Multi' if self.use_multi_gpu else 'Single'} GPU setup: using {self.device}")
            else:
                self.device = self.secondary_device = torch.device("cpu")
                self.use_multi_gpu = False
                logger.info("Using CPU")
        except ImportError:
            self.device = self.secondary_device = None
            self.use_multi_gpu = False
            logger.warning("PyTorch unavailable, GPU disabled")
    
    def _truncate_text(self, text: str, max_chars: int = 212) -> str:
        """Truncate or wrap text to max_chars per line."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text = re.sub(r'\s+', ' ', text.strip())
        if len(text) <= max_chars:
            return text
        mid_point = max_chars
        for i in range(max(0, mid_point - 20), min(len(text), mid_point + 20)):
            if text[i] == ' ':
                mid_point = i
                break
        line1 = text[:mid_point].strip()
        line2 = text[mid_point:max_chars * 2].strip()
        if len(line2) > max_chars:
            line2 = line2[:max_chars - 3] + "..."
        return f"{line1}<br>{line2}"
    
    def load_domain_mapping(self, file_path: str = "data/domains/reorganized_domain_tiny_ids.csv") -> pd.DataFrame:
        """Load and deduplicate domain mapping data."""
        print(f"Loading domain mapping from {file_path}...")
        logger.info(f"Loading domain mapping from {file_path}")
        try:
            domain_df = pd.read_csv(file_path)
            initial_count = len(domain_df)
            duplicate_mask = domain_df.duplicated(subset=['tiny_ids'], keep='first')
            if duplicate_mask.sum() > 0:
                domain_df = domain_df[~duplicate_mask].copy()
                logger.info(f"Removed {duplicate_mask.sum()} duplicates: {initial_count} -> {len(domain_df)}")
            print(f"Loaded {len(domain_df)} domain mappings, {domain_df['domain'].nunique()} unique domains")
            logger.info(f"Loaded {len(domain_df)} mappings, {domain_df['domain'].nunique()} unique domains")
            self.domain_mapping = domain_df
            return domain_df
        except Exception as e:
            print(f"Error loading domain mapping: {e}")
            logger.error(f"Error loading domain mapping: {e}")
            return pd.DataFrame()
    
    def load_cde_data(self, file_path: str = "data/cde_all_nofreqphrase_embeddingText_20250723.json") -> pd.DataFrame:
        """Load and filter CDE data by curated domain tiny_ids."""
        print(f"Loading CDE data from {file_path}...")
        logger.info(f"Loading CDE data from {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                cde_data = json.load(f)
            df = pd.DataFrame(cde_data)
            print(f"Loaded {len(df)} total CDEs")
            logger.info(f"Loaded {len(df)} CDEs")
            if self.domain_mapping is not None and 'tinyId' in df.columns:
                curated_tiny_ids = set(self.domain_mapping['tiny_ids'].values)
                df['tinyId_clean'] = df['tinyId'].astype(str).str.strip()
                curated_tiny_ids_clean = {str(tid).strip() for tid in curated_tiny_ids}
                filtered_df = df[df['tinyId_clean'].isin(curated_tiny_ids_clean)].copy()
                print(f"Filtered to {len(filtered_df)} CDEs matching curated domains")
                logger.info(f"Filtered to {len(filtered_df)} CDEs")
                self.cde_data = filtered_df
                return filtered_df
            print("No domain mapping or tinyId column found")
            logger.error("No domain mapping or tinyId column")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading CDE data: {e}")
            logger.error(f"Error loading CDE data: {e}")
            return pd.DataFrame()
    
    def extract_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract and concatenate text fields from CDE data"""
        print("Extracting text fields and adding domain labels...")
        logger.info("Extracting text fields")
        if len(df) == 0:
            print("No CDE data to process")
            return pd.DataFrame()
        processed_df = df.copy()
        processed_df['name'] = ''
        processed_df['question'] = ''
        processed_df['definition'] = ''
        processed_df['permissible_values'] = ''
        processed_df['combined_text'] = ''
        processed_df['domain'] = ''
        
        for idx in tqdm(range(len(processed_df)), desc="Processing CDEs"):
            row = processed_df.iloc[idx]
            name = next((str(row[field]).strip() for field in ['Name', 'name', 'designation', 'CDE_Name'] 
                        if field in row and pd.notna(row[field]) and str(row[field]).strip()), '')
            if not name and 'designations' in row and isinstance(row['designations'], list) and row['designations']:
                name = str(row['designations'][0].get('designation', '')).strip()
            
            question = next((str(row[field]).strip() for field in ['Question', 'question', 'CDE_Question'] 
                            if field in row and pd.notna(row[field]) and str(row[field]).strip()), '')
            if not question and 'designations' in row and isinstance(row['designations'], list) and len(row['designations']) > 1:
                question = str(row['designations'][1].get('designation', '')).strip()
            
            definition = next((str(row[field]).strip() for field in ['Definition', 'definition', 'CDE_Definition'] 
                             if field in row and pd.notna(row[field]) and str(row[field]).strip()), '')
            if not definition and 'definitions' in row and isinstance(row['definitions'], list) and row['definitions']:
                definition = str(row['definitions'][0].get('definition', '')).strip()
            
            permissible_values = next((str(row[field]).strip() for field in ['PermissibleValues.permissibleValue', 
                                                                            'permissible_values', 'permissibleValues', 
                                                                            'Permissible Values', 'permissible_values_text', 
                                                                            'PermissibleValues'] 
                                      if field in row and pd.notna(row[field]) and str(row[field]).strip()), '')
            
            name = self._clean_text(name)
            question = self._clean_text(question)
            definition = self._clean_text(definition)
            permissible_values = self._clean_text(permissible_values)
            combined_text = f"{name} {question} {definition} {permissible_values}".strip()
            
            tiny_id = str(row.get('tinyId', '')).strip()
            domain_label = 'Unknown'
            if self.domain_mapping is not None:
                domain_match = self.domain_mapping[self.domain_mapping['tiny_ids'].astype(str).str.strip() == tiny_id]
                if not domain_match.empty:
                    domain_label = domain_match.iloc[0]['domain']
            
            processed_df.iloc[idx, processed_df.columns.get_loc('name')] = name
            processed_df.iloc[idx, processed_df.columns.get_loc('question')] = question
            processed_df.iloc[idx, processed_df.columns.get_loc('definition')] = definition
            processed_df.iloc[idx, processed_df.columns.get_loc('permissible_values')] = permissible_values
            processed_df.iloc[idx, processed_df.columns.get_loc('combined_text')] = combined_text
            processed_df.iloc[idx, processed_df.columns.get_loc('domain')] = domain_label
        
        processed_df = processed_df[processed_df['combined_text'].str.len() >= 5]
        print(f"Processed {len(processed_df)} CDEs")
        print(f"Domain distribution:\n{processed_df['domain'].value_counts()}")
        logger.info(f"Processed {len(processed_df)} CDEs, domain distribution: {processed_df['domain'].value_counts().to_dict()}")
        return processed_df
    
    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        return text.strip()
    
    def load_embedding_models(self) -> None:
        print("Loading embedding models...")
        logger.info("Loading embedding models")
        try:
            self.embedding_models['all-MiniLM-L6-v2'] = SentenceTransformer('all-MiniLM-L6-v2')
            print("Loaded all-MiniLM-L6-v2")
            logger.info("Loaded all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Failed to load all-MiniLM-L6-v2: {e}")
            logger.error(f"Failed to load all-MiniLM-L6-v2: {e}")
        
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizers['SAPBERT'] = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
            self.embedding_models['SAPBERT'] = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
            print("Loaded SAPBERT")
            logger.info("Loaded SAPBERT")
        except Exception as e:
            print(f"Failed to load SAPBERT: {e}")
            logger.error(f"Failed to load SAPBERT: {e}")
        
        try:
            self.tokenizers['MedCPT'] = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
            self.embedding_models['MedCPT'] = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")
            print("Loaded MedCPT")
            logger.info("Loaded MedCPT")
        except Exception as e:
            print(f"Failed to load MedCPT: {e}")
            logger.error(f"Failed to load MedCPT: {e}")
    
    def compute_embeddings_sentence_transformer(self, model, texts: List[str]) -> np.ndarray:
        print(f"Computing embeddings for {len(texts)} texts with all-MiniLM-L6-v2...")
        logger.info(f"Computing embeddings for {len(texts)} texts")
        start_time = time.time()
        processed_texts = [re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', str(text).strip() or "empty text") for text in texts]
        try:
            embeddings = model.encode(processed_texts, show_progress_bar=True)
            print(f"Completed all-MiniLM-L6-v2 embeddings in {time.time() - start_time:.2f} seconds, shape: {embeddings.shape}")
            logger.info(f"Computed embeddings shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            print(f"Error computing embeddings: {e}")
            logger.error(f"Error computing embeddings: {e}")
            return np.zeros((len(texts), 384))
    
    def compute_embeddings_transformers(self, model, tokenizer, texts: List[str]) -> np.ndarray:
        """Compute embeddings - GPU preferred"""
        print(f"Computing embeddings for {len(texts)} texts with SAPBERT...")
        logger.info(f"Computing transformers embeddings for {len(texts)} texts")
        import torch
        start_time = time.time()
        if self.device and self.device.type == 'cuda':
            model = model.to(self.device)
        processed_texts = [re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', str(text).strip() or "empty text") for text in texts]
        batch_size = 64 if self.device and self.device.type == 'cuda' else 32
        all_embeddings = []
        
        for i in tqdm(range(0, len(processed_texts), batch_size), desc="Processing SAPBERT batches"):
            batch_texts = processed_texts[i:i+batch_size]
            try:
                encoded = tokenizer(batch_texts, truncation=True, padding=True, return_tensors='pt', max_length=512)
                if self.device and self.device.type == 'cuda':
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}
                with torch.no_grad():
                    output = model(**encoded)
                    cls_embeddings = output[0][:, 0, :].cpu().numpy()
                    all_embeddings.append(cls_embeddings)
            except Exception as e:
                logger.warning(f"Error in batch {i//batch_size + 1}: {e}")
                all_embeddings.append(np.zeros((len(batch_texts), 768)))
        
        embeddings = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.zeros((len(texts), 768))
        print(f"Completed SAPBERT embeddings in {time.time() - start_time:.2f} seconds, shape: {embeddings.shape}")
        logger.info(f"Computed embeddings shape: {embeddings.shape}")
        return embeddings
    
    def compute_embeddings_medcpt(self, model, tokenizer, texts: List[str]) -> np.ndarray:
        """Compute embeddings using MedCPT"""
        print(f"Computing embeddings for {len(texts)} texts with MedCPT...")
        logger.info(f"Computing MedCPT embeddings for {len(texts)} texts")
        import torch
        start_time = time.time()
        target_device = self.secondary_device if self.use_multi_gpu else self.device
        if target_device and target_device.type == 'cuda':
            model = model.to(target_device)
        processed_texts = [re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', ' ', str(text).strip() or "empty text") for text in texts]
        batch_size = 32 if target_device and target_device.type == 'cuda' else 16
        all_embeddings = []
        
        for i in tqdm(range(0, len(processed_texts), batch_size), desc="Processing MedCPT batches"):
            batch_texts = processed_texts[i:i+batch_size]
            try:
                encoded = tokenizer(batch_texts, truncation=True, padding=True, return_tensors='pt', max_length=128)
                if target_device and target_device.type == 'cuda':
                    encoded = {k: v.to(target_device) for k, v in encoded.items()}
                with torch.no_grad():
                    output = model(**encoded)
                    cls_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
                    all_embeddings.append(cls_embeddings)
            except Exception as e:
                logger.warning(f"Error in batch {i//batch_size + 1}: {e}")
                all_embeddings.append(np.zeros((len(batch_texts), 768)))
        
        embeddings = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.zeros((len(texts), 768))
        print(f"Completed MedCPT embeddings in {time.time() - start_time:.2f} seconds, shape: {embeddings.shape}")
        logger.info(f"MedCPT embeddings shape: {embeddings.shape}")
        return embeddings
    
    def compute_embeddings(self, model_name: str, texts: List[str]) -> np.ndarray:
        if model_name == 'all-MiniLM-L6-v2':
            return self.compute_embeddings_sentence_transformer(self.embedding_models[model_name], texts)
        elif model_name == 'SAPBERT':
            return self.compute_embeddings_transformers(self.embedding_models[model_name], self.tokenizers[model_name], texts)
        elif model_name == 'MedCPT':
            return self.compute_embeddings_medcpt(self.embedding_models[model_name], self.tokenizers[model_name], texts)
        raise ValueError(f"Unknown model: {model_name}")
    
    def apply_dimensionality_reduction(self, embeddings: np.ndarray, method: str = 'tsne') -> np.ndarray:
        """Apply t-SNE or UMAP for visualization"""
        print(f"Applying {method.upper()}...")
        logger.info(f"Applying {method} reduction")
        start_time = time.time()
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) // 4), metric='cosine', method='exact')
        elif method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(embeddings) // 3), min_dist=0.1, metric='cosine')
        else:
            raise ValueError(f"Unknown method: {method}")
        reduced_embeddings = reducer.fit_transform(embeddings_scaled)
        print(f"Completed {method.upper()} in {time.time() - start_time:.2f} seconds, shape: {reduced_embeddings.shape}")
        logger.info(f"{method} reduction shape: {reduced_embeddings.shape}")
        return reduced_embeddings
    
    def apply_clustering(self, embeddings: np.ndarray) -> Tuple[np.ndarray, HDBSCAN]:
        """HDBSCAN clustering"""
        print("Applying HDBSCAN clustering...")
        logger.info(f"Clustering embeddings shape: {embeddings.shape}")
        start_time = time.time()
        clusterer = HDBSCAN(**self.hdbscan_params)
        cluster_labels = clusterer.fit_predict(embeddings)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        print(f"Completed clustering: {n_clusters} clusters, {n_noise} noise points in {time.time() - start_time:.2f} seconds")
        logger.info(f"Clusters: {n_clusters}, Noise: {n_noise}, Coverage: {(len(cluster_labels) - n_noise) / len(cluster_labels):.4f}")
        return cluster_labels, clusterer
    
    def evaluate_clustering(self, embeddings: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering quality"""
        valid_mask = cluster_labels != -1
        n_total = len(cluster_labels)
        n_clustered = np.sum(valid_mask)
        coverage = n_clustered / n_total
        if not np.any(valid_mask) or len(set(cluster_labels[valid_mask])) < 2:
            return {'silhouette_score': 0.0, 'coverage': coverage, 'n_clusters': 0, 'n_noise': n_total - n_clustered, 'avg_cluster_size': 0.0}
        valid_embeddings = embeddings[valid_mask]
        valid_labels = cluster_labels[valid_mask]
        silhouette = silhouette_score(valid_embeddings, valid_labels)
        cluster_sizes = np.bincount(valid_labels)
        return {
            'silhouette_score': silhouette,
            'coverage': coverage,
            'n_clusters': len(set(valid_labels)),
            'n_noise': n_total - n_clustered,
            'avg_cluster_size': cluster_sizes.mean()
        }
    
    def _get_color_and_shape(self, category_idx: int) -> Tuple[str, str]:
        """Get color and marker shape for category"""
        color = self.d3_colors[category_idx % len(self.d3_colors)]
        shape = self.marker_shapes[category_idx // len(self.d3_colors) % len(self.marker_shapes)]
        return color, shape
    
    def run_analysis(self, model_name: str = 'all-MiniLM-L6-v2') -> Dict[str, Any]:
        """Run clustering analysis with t-SNE and UMAP"""
        print(f"\nStarting analysis with {model_name}...")
        logger.info(f"Running analysis with {model_name}")
        try:
            import torch
            torch.manual_seed(42)
        except ImportError:
            logger.warning("PyTorch unavailable")
        np.random.seed(42)
        
        if not self.embedding_models:
            self.load_embedding_models()
        if model_name not in self.embedding_models or self.filtered_cdes is None or len(self.filtered_cdes) == 0:
            print(f"Model {model_name} or data unavailable")
            logger.error(f"Model {model_name} or data unavailable")
            return {}
        
        texts = self.filtered_cdes['combined_text'].tolist()
        embeddings = self.compute_embeddings(model_name, texts)
        visualization_methods = ['tsne', 'umap']
        visualization_embeddings = {}
        clustering_results = {}
        self.all_metrics[model_name] = {}
        
        for method in visualization_methods:
            try:
                vis_embeddings = self.apply_dimensionality_reduction(embeddings, method)
                cluster_labels, clusterer = self.apply_clustering(vis_embeddings)
                visualization_embeddings[method] = vis_embeddings
                clustering_results[method] = {'cluster_labels': cluster_labels, 'clusterer': clusterer}
                self.all_metrics[model_name][method] = self.evaluate_clustering(vis_embeddings, cluster_labels)
                print(f"Completed {method.upper()} analysis: {self.all_metrics[model_name][method]['n_clusters']} clusters")
            except Exception as e:
                print(f"Failed {method.upper()} analysis: {e}")
                logger.warning(f"Failed {method} analysis: {e}")
        
        print(f"Analysis complete for {model_name}")
        return {
            'model_name': model_name,
            'filtered_cdes': self.filtered_cdes,
            'original_embeddings': embeddings,
            'visualization_embeddings': visualization_embeddings,
            'clustering_results': clustering_results
        }
    
    def create_faceted_plots(self, results: Dict[str, Any]) -> List[go.Figure]:
        """Create faceted t-SNE/UMAP plots for SAPBERT/MedCPT"""
        if not results or results['model_name'] not in ['SAPBERT', 'MedCPT']:
            return []
        df = results['filtered_cdes']
        visualization_embeddings = results['visualization_embeddings']
        clustering_results = results['clustering_results']
        methods = list(visualization_embeddings.keys())
        if len(methods) < 2:
            return []
        return [self._create_faceted_comparison_figure(df, visualization_embeddings, clustering_results, results['model_name'])]
    
    def _create_faceted_comparison_figure(self, df: pd.DataFrame, visualization_embeddings: Dict, 
                                        clustering_results: Dict, model_name: str) -> go.Figure:
        methods = list(visualization_embeddings.keys())
        fig = make_subplots(rows=1, cols=2, subplot_titles=[method.upper().replace('_', ' ') for method in methods],
                           specs=[[{"type": "scatter"}, {"type": "scatter"}]], horizontal_spacing=0.08)
        
        domains = df['domain'].unique()
        for method_idx, method in enumerate(methods):
            embeddings = visualization_embeddings[method]
            cluster_labels = clustering_results[method]['cluster_labels'] if method in clustering_results else np.zeros(len(df))
            for dom_idx, domain in enumerate(domains):
                mask = df['domain'] == domain
                color, shape = self._get_color_and_shape(dom_idx)
                hover_template = (
                    "<b>TinyID:</b> %{customdata[0]}<br>"
                    "<b>Domain:</b> " + domain + "<br>"
                    "<b>Cluster:</b> %{customdata[1]}<br>"
                    "<b>Name:</b><br>%{customdata[2]}<br>"
                    "<b>Question:</b><br>%{customdata[3]}<br>"
                    "<b>Definition:</b><br>%{customdata[4]}<br>"
                    "<extra></extra>"
                )
                fig.add_trace(
                    go.Scatter(
                        x=embeddings[mask, 0],
                        y=embeddings[mask, 1],
                        mode='markers',
                        name=domain if method_idx == 0 else None,
                        marker=dict(color=color, size=6, opacity=0.6, symbol=shape, line=dict(width=1, color='rgba(255,255,255,0.6)')),
                        customdata=np.column_stack([
                            df[mask]['tinyId'].values,
                            cluster_labels[mask],
                            df[mask]['name'].apply(lambda x: self._truncate_text(x, 200)).values,
                            df[mask]['question'].apply(lambda x: self._truncate_text(x, 200)).values,
                            df[mask]['definition'].apply(lambda x: self._truncate_text(x, 200)).values
                        ]),
                        hovertemplate=hover_template,
                        showlegend=(method_idx == 0),
                        legendgroup=domain,
                        selected=dict(marker=dict(size=10, opacity=0.8)),
                        unselected=dict(marker=dict(opacity=0.4))
                    ),
                    row=1, col=method_idx + 1
                )
        
        fig.update_layout(
            title=f'Interactive {model_name} Clustering Analysis - t-SNE vs UMAP',
            width=1600, height=700, font=dict(size=12, family="Arial"), title_font_size=18,
            template='plotly_white', showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font=dict(size=12),
                       bgcolor='rgba(255,255,255,0.8)', bordercolor='rgba(0,0,0,0.2)', borderwidth=1),
            margin=dict(l=50, r=50, t=100, b=100)
        )
        for i in range(1, 3):
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', row=1, col=i, title_font_size=12)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', row=1, col=i, title_font_size=12)
        return fig
    
    def create_dash_app(self) -> dash.Dash:
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        app.layout = dbc.Container([
            dbc.Row([dbc.Col([html.H1("Interactive CDE Clustering Analysis", className="text-center mb-4"), html.Hr()])]),
            dbc.Row([dbc.Col([dbc.Card([
                dbc.CardHeader("Model Selection"),
                dbc.CardBody([dbc.RadioItems(id='model-selector', options=[
                    {'label': 'SAPBERT', 'value': 'SAPBERT'}, {'label': 'MedCPT', 'value': 'MedCPT'}],
                    value='SAPBERT', inline=True), html.Div(id='model-status', className="mt-2")])])])], className="mb-4"),
            dbc.Row([dbc.Col([dbc.Card([
                dbc.CardHeader("Interactive Clustering Visualization"),
                dbc.CardBody([dcc.Graph(id='clustering-plot', style={'height': '700px'},
                                       config={'displayModeBar': True, 'modeBarButtonsToAdd': ['select2d', 'lasso2d', 'resetScale2d'],
                                              'displaylogo': False})])])])], className="mb-4"),
            dbc.Row([dbc.Col([dbc.Card([
                dbc.CardHeader("Data Export"),
                dbc.CardBody([
                    dbc.Row([dbc.Col([html.P("Select data points using lasso or box selection tools above, then export:"),
                                     html.Div(id='selection-info', className="mb-3")])]),
                    dbc.Row([dbc.Col([
                        dbc.Button("Export to JSON", id='export-json-btn', color="primary", className="me-2"),
                        dbc.Button("Export to CSV", id='export-csv-btn', color="success", className="me-2"),
                        dbc.Button("Copy to Clipboard", id='copy-clipboard-btn', color="info")])]),
                    dbc.Row([dbc.Col([html.Div(id='export-status', className="mt-3")])])
                ])])])]),
            html.Div(id='selected-data-store', style={'display': 'none'}),
            html.Div(id='current-model-store', style={'display': 'none'})
        ], fluid=True)
        self.app = app
        return app
    
    def setup_callbacks(self):
        """Setup Dash callbacks"""
        if not self.app:
            logger.error("Dash app not initialized")
            return
        
        @self.app.callback(
            [Output('clustering-plot', 'figure'), Output('current-model-store', 'children')],
            [Input('model-selector', 'value')]
        )
        def update_plot(selected_model):
            """Update plot based on model selection"""
            print(f"Generating plot for {selected_model}...")
            if selected_model not in self.embedding_models:
                empty_fig = go.Figure()
                empty_fig.add_annotation(text=f"Model '{selected_model}' not available.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="red"))
                empty_fig.update_layout(title=f"Model Not Available: {selected_model}", width=800, height=400)
                return empty_fig, selected_model
            if selected_model not in self.analysis_results:
                results = self.run_analysis(selected_model)
                if results:
                    self.analysis_results[selected_model] = results
                    figures = self.create_faceted_plots(results)
                    if figures:
                        print(f"Plot generated for {selected_model}")
                        return figures[0], selected_model
            if selected_model in self.analysis_results:
                figures = self.create_faceted_plots(self.analysis_results[selected_model])
                if figures:
                    print(f"Plot generated for {selected_model}")
                    return figures[0], selected_model
            empty_fig = go.Figure()
            empty_fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            print(f"No data for {selected_model}")
            return empty_fig, selected_model
        
        @self.app.callback(
            [Output('selection-info', 'children'), Output('selected-data-store', 'children')],
            [Input('clustering-plot', 'selectedData')]
        )
        def update_selection_info(selected_data):
            """Update selection info and store data."""
            if not selected_data or not selected_data.get('points'):
                return "No points selected", ""
            points = selected_data['points']
            selected_data_list = [
                {'tiny_id': point['customdata'][0], 'cluster': point['customdata'][1], 'name': point['customdata'][2],
                 'question': point['customdata'][3], 'definition': point['customdata'][4], 'domain': point.get('legendgroup', 'Unknown'),
                 'x': point['x'], 'y': point['y']} for point in points if 'customdata' in point]
            return f"Selected {len(points)} data point(s)", json.dumps(selected_data_list)
        
        @self.app.callback(
            Output('export-status', 'children'),
            [Input('export-json-btn', 'n_clicks'), Input('export-csv-btn', 'n_clicks'), Input('copy-clipboard-btn', 'n_clicks')],
            [State('selected-data-store', 'children'), State('current-model-store', 'children')]
        )
        def handle_export(json_clicks, csv_clicks, clipboard_clicks, selected_data_json, current_model):
            """Handle data export."""
            ctx = callback_context
            if not ctx.triggered or not selected_data_json:
                return dbc.Alert("No data selected", color="warning")
            try:
                selected_data = json.loads(selected_data_json)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                if ctx.triggered[0]['prop_id'].split('.')[0] == 'export-json-btn':
                    filename = f"selected_cdes_{current_model}_{timestamp}.json"
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(selected_data, f, indent=2, ensure_ascii=False)
                    return dbc.Alert(f"Data exported to {filename}", color="success")
                elif ctx.triggered[0]['prop_id'].split('.')[0] == 'export-csv-btn':
                    filename = f"selected_cdes_{current_model}_{timestamp}.csv"
                    pd.DataFrame(selected_data).to_csv(filename, index=False, encoding='utf-8')
                    return dbc.Alert(f"Data exported to {filename}", color="success")
                elif ctx.triggered[0]['prop_id'].split('.')[0] == 'copy-clipboard-btn':
                    pyperclip.copy(self._format_clipboard_data(selected_data))
                    return dbc.Alert("Data copied to clipboard", color="success")
            except Exception as e:
                logger.error(f"Export error: {e}")
                return dbc.Alert(f"Export failed: {str(e)}", color="danger")
            return ""
        
        @self.app.callback(
            Output('model-status', 'children'),
            [Input('model-selector', 'value')]
        )
        def update_model_status(selected_model):
            return dbc.Alert(f" {selected_model} loaded" if selected_model in self.embedding_models else f" {selected_model} unavailable", 
                            color="success" if selected_model in self.embedding_models else "danger")
    
    def _format_clipboard_data(self, selected_data: List[Dict]) -> str:
        if not selected_data:
            return "No data selected"
        lines = ["Selected CDE Data", "=" * 50, ""]
        for i, item in enumerate(selected_data, 1):
            lines.extend([
                f"CDE {i}:", f"  Tiny ID: {item.get('tiny_id', 'N/A')}", f"  Domain: {item.get('domain', 'N/A')}",
                f"  Cluster: {item.get('cluster', 'N/A')}", f"  Name: {item.get('name', 'N/A')}",
                f"  Question: {item.get('question', 'N/A')}", f"  Definition: {item.get('definition', 'N/A')}",
                f"  Coordinates: ({item.get('x', 'N/A'):.3f}, {item.get('y', 'N/A'):.3f})", ""
            ])
        return "\n".join(lines)

def main():
    analyzer = InteractiveClusteringAnalyzer(hdbscan_min_cluster_size=15, hdbscan_min_samples=5)
    domain_df = analyzer.load_domain_mapping()
    if domain_df.empty:
        print("Error: No domain mapping loaded")
        return
    cde_df = analyzer.load_cde_data()
    if cde_df.empty:
        print("Error: No CDE data loaded")
        return
    processed_df = analyzer.extract_text_fields(cde_df)
    if processed_df.empty:
        print("Error: No processed CDE data")
        return
    analyzer.filtered_cdes = processed_df
    try:
        analyzer.load_embedding_models()
        print("All embedding models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
    print("Creating Dash application...")
    app = analyzer.create_dash_app()
    print("Setting up interactive callbacks...")
    analyzer.setup_callbacks()
    print("\nInteractive Clustering Analysis Ready")
    # print("Open browser to: http://127.0.0.1:8050")
    try:
        app.run(debug=False, host='127.0.0.1', port=8050)
    except KeyboardInterrupt:
        print("Session ended by user")
    except Exception as e:
        logger.error(f"Dash server error: {e}")
        print(f"Error running Dash server: {e}")

if __name__ == "__main__":
    main()