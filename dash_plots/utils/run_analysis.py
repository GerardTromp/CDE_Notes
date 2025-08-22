import umap  # type: ignore
import numpy as np
import time
from tqdm import tqdm
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN  # type: ignore
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from utils.functions import logger, date_time_string


def apply_dimensionality_reduction(
    self, embeddings: np.ndarray, method: str = "tsne"
) -> np.ndarray:
    """Apply t-SNE or UMAP for visualization"""
    print(f"Applying {method.upper()}...")
    logger.info(f"Applying {method} reduction")
    start_time = time.time()
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    # print(f"{self.analysis_params}\n")
    # tsne_params = self.analysis_params["TSNE"]
    # umap_params = self.analysis_params["UMAP"]
    tsne_params = self.tsne_params
    umap_params = self.umap_params
    # print(f"modified TSNE params {tsne_params}\n")
    # print(f"modified UMAP params {umap_params}\n")
    ####
    # Move the parameter update out of this function so that the dash app can dynamically update the dictionary
    tsne_local = {
        "n_components": 2,
        "random_state": 42,
        "perplexity": min(30, len(embeddings) // 4),
        "metric": "cosine",
        "method": "exact",
    }
    tsne_params.update(tsne_local)
    umap_local = {
        "n_components": 2,
        "random_state": 42,
        "n_neighbors": min(15, len(embeddings) // 3),
        "min_dist": 0.1,
        "metric": "cosine",
    }
    umap_params.update(umap_local)
    if method == "tsne":
        # reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) // 4), metric='cosine', method='exact')
        reducer = TSNE(**tsne_params)
    elif method == "umap":
        # reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(embeddings) // 3), min_dist=0.1, metric='cosine')
        reducer = umap.UMAP(**umap_params)
    else:
        raise ValueError(f"Unknown method: {method}")
    reduced_embeddings = reducer.fit_transform(embeddings_scaled)
    print(
        f"Completed {method.upper()} in {time.time() - start_time:.2f} seconds, shape: {reduced_embeddings.shape}"
    )
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
    print(
        f"Completed clustering: {n_clusters} clusters, {n_noise} noise points in {time.time() - start_time:.2f} seconds"
    )
    logger.info(
        f"Clusters: {n_clusters}, Noise: {n_noise}, Coverage: {(len(cluster_labels) - n_noise) / len(cluster_labels):.4f}"
    )
    return cluster_labels, clusterer


def evaluate_clustering(
    self, embeddings: np.ndarray, cluster_labels: np.ndarray
) -> Dict[str, float]:
    """Evaluate clustering quality"""
    valid_mask = cluster_labels != -1
    n_total = len(cluster_labels)
    n_clustered = np.sum(valid_mask)
    coverage = n_clustered / n_total
    if not np.any(valid_mask) or len(set(cluster_labels[valid_mask])) < 2:
        return {
            "silhouette_score": 0.0,
            "coverage": coverage,
            "n_clusters": 0,
            "n_noise": n_total - n_clustered,
            "avg_cluster_size": 0.0,
        }
    valid_embeddings = embeddings[valid_mask]
    valid_labels = cluster_labels[valid_mask]
    silhouette = silhouette_score(valid_embeddings, valid_labels)
    cluster_sizes = np.bincount(valid_labels)
    return {
        "silhouette_score": silhouette,  # type: ignore
        "coverage": coverage,
        "n_clusters": len(set(valid_labels)),
        "n_noise": n_total - n_clustered,
        "avg_cluster_size": cluster_sizes.mean(),
    }


##################################
# This should be generalized to permit any model to be run
#    Changing to load precomputed embeddings
#    Change design to separate embedding and analysis
#
def run_analysis(self, model_name: str = "all-MiniLM-L6-v2") -> Dict[str, Any]:
    """Run clustering analysis with t-SNE and UMAP"""
    print(f"\nStarting analysis with {model_name}...")
    logger.info(f"Running analysis with {model_name}")
    # try:
    #     import torch

    #     torch.manual_seed(42)
    # except ImportError:
    #     logger.warning("PyTorch unavailable")
    np.random.seed(42)

    if not self.embedding_models:
        self.load_embedding_models()
    if (
        model_name not in self.embedding_models
        or self.filtered_cdes is None
        or len(self.filtered_cdes) == 0
    ):
        print(f"Model {model_name} or data unavailable")
        logger.error(f"Model {model_name} or data unavailable")
        return {}

    texts = self.filtered_cdes["combined_text"].tolist()
    # embeddings = self.compute_embeddings(model_name, texts)
    embeddings = self.embedding_models[model_name]
    visualization_methods = ["tsne", "umap"]
    visualization_embeddings = {}
    clustering_results = {}
    self.all_metrics[model_name] = {}

    for method in visualization_methods:
        try:
            vis_embeddings = self.apply_dimensionality_reduction(embeddings, method)
            cluster_labels, clusterer = self.apply_clustering(vis_embeddings)
            visualization_embeddings[method] = vis_embeddings
            clustering_results[method] = {
                "cluster_labels": cluster_labels,
                "clusterer": clusterer,
            }
            self.all_metrics[model_name][method] = self.evaluate_clustering(
                vis_embeddings, cluster_labels
            )
            print(
                f"Completed {method.upper()} analysis: {self.all_metrics[model_name][method]['n_clusters']} clusters"
            )
        except Exception as e:
            print(f"Failed {method.upper()} analysis: {e}")
            logger.warning(f"Failed {method} analysis: {e}")

    print(f"Analysis complete for {model_name}")
    return {
        "model_name": model_name,
        "filtered_cdes": self.filtered_cdes,
        "original_embeddings": embeddings,
        "visualization_embeddings": visualization_embeddings,
        "clustering_results": clustering_results,
    }
