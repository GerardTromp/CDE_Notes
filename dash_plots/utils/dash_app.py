import dash  # type: ignore
import dash_bootstrap_components as dbc  # type: ignore
import json
import pandas as pd  # type: ignore
import numpy as np
import pyperclip  # type: ignore
import plotly.graph_objects as go  # type: ignore
from dash import dcc, html, Input, Output, State, callback_context  # type: ignore
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from utils.functions import logger


def create_dash_app(self) -> dash.Dash:
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    models = self.embedding_models.keys()
    model_options = []
    for model in models:
        print(model)
        model_options.append({"label": model, "value": model})
    value_option = model_options[0]["value"]
    app.layout = dbc.Container(
        [
            # fmt: off
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H1( "Interactive CDE Clustering Analysis", className="text-center mb-4",),
                            html.Hr(),
                        ]
                    )
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Model Selection"),
                                    dbc.CardBody(
                                        [
                                            dbc.RadioItems( id="model-selector", options=model_options, value=value_option, inline=True,),
                                            html.Div( id="model-status", className="mt-2"),
                                        ]
                                    ),
                                ]
                            )
                        ]
                    )
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        "Interactive Clustering Visualization"
                                    ),
                                    dbc.CardBody(
                                        [
                                            dcc.Graph(
                                                id="clustering-plot",
                                                style={"height": "700px"},
                                                config={ "displayModeBar": True, "modeBarButtonsToAdd": [ "select2d", "lasso2d", "resetScale2d", ],
                                                    "displaylogo": False,
                                                },
                                            )
                                        ]
                                    ),
                                ]
                            )
                        ]
                    )
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Data Export"),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.P( "Select data points using lasso or box selection tools above, then export:"),
                                                            html.Div( id="selection-info", className="mb-3",),
                                                        ]
                                                    )
                                                ]
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            dbc.Button( "Export to JSON", id="export-json-btn", color="primary", className="me-2",),
                                                            dbc.Button( "Export to CSV", id="export-csv-btn", color="success", className="me-2",),
                                                            dbc.Button( "Copy to Clipboard", id="copy-clipboard-btn", color="info",),
                                                        ]
                                                    )
                                                ]
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        [
                                                            html.Div( id="export-status", className="mt-3",
                                                            )
                                                        ]
                                                    )
                                                ]
                                            ),
                                        ]
                                    ),
                                ]
                            )
                        ]
                    )
                ]
            ),
            # fmt: on
            html.Div(id="selected-data-store", style={"display": "none"}),
            html.Div(id="current-model-store", style={"display": "none"}),
        ],
        fluid=True,
    )
    self.app = app
    return app


def setup_callbacks(self):
    """Setup Dash callbacks"""
    if not self.app:
        logger.error("Dash app not initialized")
        return

    @self.app.callback(
        [
            Output("clustering-plot", "figure"),
            Output("current-model-store", "children"),
        ],
        [Input("model-selector", "value")],
    )
    def update_plot(selected_model):
        """Update plot based on model selection"""
        print(f"Generating plot for {selected_model}...")
        if selected_model not in self.embedding_models:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text=f"Model '{selected_model}' not available.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="red"),
            )
            empty_fig.update_layout(
                title=f"Model Not Available: {selected_model}",
                width=800,
                height=400,
            )
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
        empty_fig.add_annotation(
            text="No data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        print(f"No data for {selected_model}")
        return empty_fig, selected_model

    @self.app.callback(
        [
            Output("selection-info", "children"),
            Output("selected-data-store", "children"),
        ],
        [Input("clustering-plot", "selectedData")],
    )
    def update_selection_info(selected_data):
        """Update selection info and store data."""
        if not selected_data or not selected_data.get("points"):
            return "No points selected", ""
        points = selected_data["points"]
        selected_data_list = [
            {
                "tinyid": point["customdata"][0],
                "cluster": point["customdata"][1],
                "name": point["customdata"][2],
                "question": point["customdata"][3],
                "definition": point["customdata"][4],
                "domain": point.get("legendgroup", "Unknown"),
                "x": point["x"],
                "y": point["y"],
            }
            for point in points
            if "customdata" in point
        ]
        return f"Selected {len(points)} data point(s)", json.dumps(selected_data_list)

    @self.app.callback(
        Output("export-status", "children"),
        [
            Input("export-json-btn", "n_clicks"),
            Input("export-csv-btn", "n_clicks"),
            Input("copy-clipboard-btn", "n_clicks"),
        ],
        [
            State("selected-data-store", "children"),
            State("current-model-store", "children"),
        ],
    )
    def handle_export(
        json_clicks, csv_clicks, clipboard_clicks, selected_data_json, current_model
    ):
        """Handle data export."""
        ctx = callback_context
        if not ctx.triggered or not selected_data_json:
            return dbc.Alert("No data selected", color="warning")
        try:
            selected_data = json.loads(selected_data_json)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if ctx.triggered[0]["prop_id"].split(".")[0] == "export-json-btn":
                filename = f"selected_cdes_{current_model}_{timestamp}.json"
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(selected_data, f, indent=2, ensure_ascii=False)
                return dbc.Alert(f"Data exported to {filename}", color="success")
            elif ctx.triggered[0]["prop_id"].split(".")[0] == "export-csv-btn":
                filename = f"selected_cdes_{current_model}_{timestamp}.csv"
                pd.DataFrame(selected_data).to_csv(
                    filename, index=False, encoding="utf-8"
                )
                return dbc.Alert(f"Data exported to {filename}", color="success")
            elif ctx.triggered[0]["prop_id"].split(".")[0] == "copy-clipboard-btn":
                pyperclip.copy(self._format_clipboard_data(selected_data))
                return dbc.Alert("Data copied to clipboard", color="success")
        except Exception as e:
            logger.error(f"Export error: {e}")
            return dbc.Alert(f"Export failed: {str(e)}", color="danger")
        return ""

    @self.app.callback(
        Output("model-status", "children"), [Input("model-selector", "value")]
    )
    def update_model_status(selected_model):
        return dbc.Alert(
            (
                f" {selected_model} loaded"
                if selected_model in self.embedding_models
                else f" {selected_model} unavailable"
            ),
            color=("success" if selected_model in self.embedding_models else "danger"),
        )


def _format_clipboard_data(self, selected_data: List[Dict]) -> str:
    if not selected_data:
        return "No data selected"
    lines = ["Selected CDE Data", "=" * 50, ""]
    for i, item in enumerate(selected_data, 1):
        lines.extend(
            [
                f"CDE {i}:",
                f"  Tiny ID: {item.get('tiny_id', 'N/A')}",
                f"  Domain: {item.get('domain', 'N/A')}",
                f"  Cluster: {item.get('cluster', 'N/A')}",
                f"  Name: {item.get('name', 'N/A')}",
                f"  Question: {item.get('question', 'N/A')}",
                f"  Definition: {item.get('definition', 'N/A')}",
                f"  Coordinates: ({item.get('x', 'N/A'):.3f}, {item.get('y', 'N/A'):.3f})",
                "",
            ]
        )
    return "\n".join(lines)
