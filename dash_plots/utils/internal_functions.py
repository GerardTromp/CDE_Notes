import re
import numpy as np
import pandas as pd  # type: ignore
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go  # type: ignore
import plotly.subplots as sp  # type: ignore


def _truncate_text(self, text: str, max_chars: int = 212) -> str:
    """Truncate or wrap text to max_chars per line."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = re.sub(r"\s+", " ", text.strip())
    if len(text) <= max_chars:
        return text
    mid_point = max_chars
    for i in range(max(0, mid_point - 20), min(len(text), mid_point + 20)):
        if text[i] == " ":
            mid_point = i
            break
    line1 = text[:mid_point].strip()
    line2 = text[mid_point : max_chars * 2].strip()
    if len(line2) > max_chars:
        line2 = line2[: max_chars - 3] + "..."
    return f"{line1}<br>{line2}"


def _get_color_and_shape(self, category_idx: int) -> Tuple[str, str]:
    """Get color and marker shape for category"""
    color = self.d3_colors[category_idx % len(self.d3_colors)]
    shape = self.marker_shapes[
        category_idx // len(self.d3_colors) % len(self.marker_shapes)
    ]
    return color, shape


def _create_faceted_comparison_figure(
    self,
    df: pd.DataFrame,
    visualization_embeddings: Dict,
    clustering_results: Dict,
    model_name: str,
) -> go.Figure:
    methods = list(visualization_embeddings.keys())
    fig = sp.make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[method.upper().replace("_", " ") for method in methods],
        specs=[[{"type": "scatter"}, {"type": "scatter"}]],
        horizontal_spacing=0.08,
    )

    domains = df["domain"].unique()
    for method_idx, method in enumerate(methods):
        embeddings = visualization_embeddings[method]
        cluster_labels = (
            clustering_results[method]["cluster_labels"]
            if method in clustering_results
            else np.zeros(len(df))
        )
        for dom_idx, domain in enumerate(domains):
            mask = df["domain"] == domain
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
                    mode="markers",
                    name=domain if method_idx == 0 else None,
                    marker=dict(
                        color=color,
                        size=6,
                        opacity=0.6,
                        symbol=shape,
                        line=dict(width=1, color="rgba(255,255,255,0.6)"),
                    ),
                    customdata=np.column_stack(
                        [
                            df[mask]["tinyId"].values,
                            cluster_labels[mask],
                            df[mask]["name"]
                            .apply(lambda x: self._truncate_text(x, 200))
                            .values,
                            df[mask]["question"]
                            .apply(lambda x: self._truncate_text(x, 200))
                            .values,
                            df[mask]["definition"]
                            .apply(lambda x: self._truncate_text(x, 200))
                            .values,
                        ]
                    ),
                    hovertemplate=hover_template,
                    showlegend=(method_idx == 0),
                    legendgroup=domain,
                    selected=dict(marker=dict(size=10, opacity=0.8)),
                    unselected=dict(marker=dict(opacity=0.4)),
                ),
                row=1,
                col=method_idx + 1,
            )

    fig.update_layout(
        title=f"Interactive {model_name} Clustering Analysis - t-SNE vs UMAP",
        width=1600,
        height=700,
        font=dict(size=12, family="Arial"),
        title_font_size=18,
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
        margin=dict(l=50, r=50, t=100, b=100),
    )
    for i in range(1, 3):
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.2)",
            row=1,
            col=i,
            title_font_size=12,
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128,128,128,0.2)",
            row=1,
            col=i,
            title_font_size=12,
        )
    return fig


def _clean_text(self, text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[^\w\s\.\,\;\:\!\?\-\(\)]", "", text)
    return text.strip()
