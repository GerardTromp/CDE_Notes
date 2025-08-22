#!/usr/bin/env python3
"""
Interactive CDE clustering analysis with Dash and Plotly.
Data point selection with lasso/box & export tooltip data to JSON/CSV or clipboard
This version comes with faceted t-SNE/UMAP plots for SAPBERT/MedCPT
"""

# import logging
import warnings

# import time
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# from sentence_transformers import SentenceTransformer  # type: ignore
from pathlib import Path
from collections import defaultdict
from utils.functions import (
    logger,
    extract_text_fields,
    create_faceted_plots,
    load_domain_mapping,
    load_cde_data,
    load_embedding_models,
    load_configs,
)
from utils.dash_app import create_dash_app, setup_callbacks
from utils.argparse import cde_argparse
from utils.internal_functions import (
    _truncate_text,
    _get_color_and_shape,
    _create_faceted_comparison_figure,
    _clean_text,
)
from utils.run_analysis import (
    run_analysis,
    apply_clustering,
    apply_dimensionality_reduction,
    evaluate_clustering,
)

warnings.filterwarnings("ignore")

# Dash imports


class InteractiveClusteringAnalyzer:
    """Interactive CDE clustering analyzer with Dash interface."""

    def __init__(
        self, hdbscan_min_cluster_size: int = 15, hdbscan_min_samples: int = 5
    ):
        """Initialize analyzer with HDBSCAN parameters."""
        # fmt: off
        self.d3_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
            "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
            "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d3", "#c7c7c7",
            "#dbdb8d", "#9edae5",
        ]
        self.marker_shapes = [
            "circle", "square", "diamond", "cross",
            "x", "triangle-up", "triangle-down", "star",
        ]
        # fmt: on
        self.embedding_models = {}
        self.tokenizers = {}
        self.domain_mapping = None
        self.cde_data = None
        self.filtered_cdes = None
        self.all_metrics = {}
        self.analysis_results = {}
        self.config = {}
        self.hdbscan_params = {}
        self.umap_params = {}
        self.tsne_params = {}
        self.analysis_params = {}
        self.app = None

    load_domain_mapping = load_domain_mapping
    extract_text_fields = extract_text_fields
    evaluate_clustering = evaluate_clustering
    load_cde_data = load_cde_data
    create_dash_app = create_dash_app
    setup_callbacks = setup_callbacks
    load_embedding_models = load_embedding_models
    load_configs = load_configs
    _truncate_text = _truncate_text
    _get_color_and_shape = _get_color_and_shape
    _create_faceted_comparison_figure = _create_faceted_comparison_figure
    _clean_text = _clean_text
    run_analysis = run_analysis
    apply_clustering = apply_clustering
    apply_dimensionality_reduction = apply_dimensionality_reduction
    create_faceted_plots = create_faceted_plots


def main():
    args = cde_argparse()

    analyzer = InteractiveClusteringAnalyzer(
        # hdbscan_min_cluster_size=15, hdbscan_min_samples=5
    )
    config = load_configs(path=args.config_path)
    if len(config) == 0:
        print("Error: could not load configs")
        return
    else:
        analyzer.config = config
    params = load_configs(path=args.param_path)
    if len(params) == 0:
        print("Error: could not load params")
        return
    else:
        # analyzer.analysis_params = params
        analyzer.umap_params = params["UMAP"]
        analyzer.tsne_params = params["TSNE"]
        analyzer.hdbscan_params = params["HDBSCAN"]

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
    analyzer.filtered_cdes = processed_df  # type: ignore
    try:
        analyzer.load_embedding_models()
        print("All embedding models loaded successfully")
    except Exception as e:
        print(f"\n--- Error loading models: {e}---\n")
    print(
        f"These are the models loaded into 'embeddding_models': {analyzer.embedding_models.keys()}"
    )
    print("Creating Dash application...")
    app = analyzer.create_dash_app()
    print("Setting up interactive callbacks...")
    analyzer.setup_callbacks()
    print("\nInteractive Clustering Analysis Ready")
    # print("Open browser to: http://127.0.0.1:8050")
    try:
        app.run(debug=False, host="127.0.0.1", port=8050)
    except KeyboardInterrupt:
        print("Session ended by user")
    except Exception as e:
        logger.error(f"Dash server error: {e}")
        print(f"Error running Dash server: {e}")


if __name__ == "__main__":
    main()
