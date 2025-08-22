import logging
import pandas as pd  # type: ignore
import numpy as np
import json
import re
import ast
import os
import sys
import configparser
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore
import umap  # type: ignore
from tqdm import tqdm
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.cluster import HDBSCAN  # type: ignore
from sklearn.metrics import silhouette_score
from datetime import datetime


def date_time_string():
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y%m%d-%H%M%S")
    return formatted_datetime


def extract_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
    """Extract and concatenate text fields from CDE data"""
    print("Extracting text fields and adding domain labels...")
    logger.info("Extracting text fields")
    if len(df) == 0:
        print("No CDE data to process")
        return pd.DataFrame()
    processed_df = df.copy()
    processed_df["name"] = ""
    processed_df["question"] = ""
    processed_df["definition"] = ""
    processed_df["permissible_values"] = ""
    processed_df["combined_text"] = ""
    processed_df["domain"] = ""

    for idx in tqdm(range(len(processed_df)), desc="Processing CDEs"):
        row = processed_df.iloc[idx]
        name = next(
            (
                str(row[field]).strip()
                for field in ["Name", "name", "designation", "CDE_Name"]
                if field in row and pd.notna(row[field]) and str(row[field]).strip()
            ),
            "",
        )
        if (
            not name
            and "designations" in row
            and isinstance(row["designations"], list)
            and row["designations"]
        ):
            name = str(row["designations"][0].get("designation", "")).strip()

        question = next(
            (
                str(row[field]).strip()
                for field in ["Question", "question", "CDE_Question"]
                if field in row and pd.notna(row[field]) and str(row[field]).strip()
            ),
            "",
        )
        if (
            not question
            and "designations" in row
            and isinstance(row["designations"], list)
            and len(row["designations"]) > 1
        ):
            question = str(row["designations"][1].get("designation", "")).strip()

        definition = next(
            (
                str(row[field]).strip()
                for field in ["Definition", "definition", "CDE_Definition"]
                if field in row and pd.notna(row[field]) and str(row[field]).strip()
            ),
            "",
        )
        if (
            not definition
            and "definitions" in row
            and isinstance(row["definitions"], list)
            and row["definitions"]
        ):
            definition = str(row["definitions"][0].get("definition", "")).strip()

        permissible_values = next(
            (
                str(row[field]).strip()
                for field in [
                    "PermissibleValues.permissibleValue",
                    "permissible_values",
                    "permissibleValues",
                    "Permissible Values",
                    "permissible_values_text",
                    "PermissibleValues",
                ]
                if field in row and pd.notna(row[field]) and str(row[field]).strip()
            ),
            "",
        )

        name = self._clean_text(name)
        question = self._clean_text(question)
        definition = self._clean_text(definition)
        permissible_values = self._clean_text(permissible_values)
        combined_text = f"{name} {question} {definition} {permissible_values}".strip()

        tiny_id = str(row.get("tinyId", "")).strip()
        domain_label = "Unknown"
        if self.domain_mapping is not None:
            domain_match = self.domain_mapping[
                self.domain_mapping["tinyid"].astype(str).str.strip() == tiny_id
            ]
            if not domain_match.empty:
                domain_label = domain_match.iloc[0]["domain"]

        processed_df.iloc[idx, processed_df.columns.get_loc("name")] = name  # type: ignore
        processed_df.iloc[idx, processed_df.columns.get_loc("question")] = question  # type: ignore
        processed_df.iloc[idx, processed_df.columns.get_loc("definition")] = definition  # type: ignore
        processed_df.iloc[idx, processed_df.columns.get_loc("permissible_values")] = (  # type: ignore
            permissible_values
        )
        processed_df.iloc[idx, processed_df.columns.get_loc("combined_text")] = (  # type: ignore
            combined_text
        )
        processed_df.iloc[idx, processed_df.columns.get_loc("domain")] = domain_label  # type: ignore

    processed_df = processed_df[processed_df["combined_text"].str.len() >= 5]
    print(f"Processed {len(processed_df)} CDEs")
    print(f"Domain distribution:\n{processed_df['domain'].value_counts()}")
    logger.info(
        f"Processed {len(processed_df)} CDEs, domain distribution: {processed_df['domain'].value_counts().to_dict()}"
    )
    return processed_df


def create_faceted_plots(self, results: Dict[str, Any]) -> List[go.Figure]:
    """Create faceted t-SNE/UMAP plots for SAPBERT/MedCPT"""
    print(f"------------- {date_time_string()} ---------------")
    if not results or results["model_name"] not in ["sapbert", "minilm"]:
        return []
    df = results["filtered_cdes"]
    visualization_embeddings = results["visualization_embeddings"]
    clustering_results = results["clustering_results"]
    methods = list(visualization_embeddings.keys())
    if len(methods) < 2:
        return []
    return [
        self._create_faceted_comparison_figure(
            df, visualization_embeddings, clustering_results, results["model_name"]
        )
    ]


def load_domain_mapping(
    self, file_path: str = "data/domains/reorganized_domain_tiny_ids.csv"
) -> pd.DataFrame:
    """Load and deduplicate domain mapping data."""
    config = self.config["CDEAnalysis"]
    print(config)
    if config["domains"]:
        file_path = config["domains"]
        print(f"This is the file path from the config array: {file_path}\n")
    print(f"Loading domain mapping from {file_path} ...")
    logger.info(f"Loading domain mapping from {file_path}")
    try:
        if "json" in file_path:
            domain_df = pd.read_json(file_path)
        elif "csv" in file_path:
            domain_df = pd.read_csv(file_path)
        initial_count = len(domain_df)
        duplicate_mask = domain_df.duplicated(subset=["tinyid"], keep="first")
        if duplicate_mask.sum() > 0:
            domain_df = domain_df[~duplicate_mask].copy()
            logger.info(
                f"Removed {duplicate_mask.sum()} duplicates: {initial_count} -> {len(domain_df)}"
            )
        print(
            f"Loaded {len(domain_df)} domain mappings, {domain_df['domain'].nunique()} unique domains"
        )
        logger.info(
            f"Loaded {len(domain_df)} mappings, {domain_df['domain'].nunique()} unique domains"
        )
        self.domain_mapping = domain_df
        return domain_df
    except Exception as e:
        print(f"Error loading domain mapping: {e}")
        logger.error(f"Error loading domain mapping: {e}")
        return pd.DataFrame()


def load_cde_data(
    self, file_path: str = "data/cde_all_nofreqphrase_embeddingText_20250729.csv"
) -> pd.DataFrame:
    """Load and filter CDE data by curated domain tinyids."""
    print(f"Loading CDE data from {file_path}...")
    logger.info(f"Loading CDE data from {file_path}")
    try:
        # with open(file_path, "r", encoding="utf-8") as f:
        #     cde_data = json.load(f)
        # df = pd.DataFrame(cde_data)
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} total CDEs")
        logger.info(f"Loaded {len(df)} CDEs")
        if self.domain_mapping is not None and "tinyId" in df.columns:
            curated_tiny_ids = set(self.domain_mapping["tinyid"].values)
            df["tinyId_clean"] = df["tinyId"].astype(str).str.strip()
            curated_tiny_ids_clean = {str(tid).strip() for tid in curated_tiny_ids}
            filtered_df = df[df["tinyId_clean"].isin(curated_tiny_ids_clean)].copy()
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


def setup_logging():
    """Configure logging for console (WARNING+) and file (DEBUG+)."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    file_handler = logging.FileHandler("interactive_clustering.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def load_embedding_models(self) -> Dict:
    modeldata = {}
    config = self.config["CDEAnalysis"]
    for model in (
        config["models"] if isinstance(config["models"], list) else [config["models"]]
    ):
        for text in (
            config["embedtext"]
            if isinstance(config["embedtext"], list)
            else [config["embedtext"]]
        ):
            embed = load_embedding_model(self, modelname=model, embedding=text)
            if embed is not None:
                modeldata[model] = embed
    print(f"Finished loading embedding models. Keys are: {modeldata.keys()}")
    self.embedding_models = modeldata
    return modeldata


def load_embedding_model(
    self, modelname: str, embedding: str
) -> Union[NDArray[np.float64], None]:
    """
    Load precomputed embeddings. Since these are large matrixes, load one at a time and process
    to the point of subsetting
    """
    config = self.config["CDEAnalysis"]
    formatstr = config["template"]
    # print(f"In the function load_embedding_model. format string is: {formatstr}")
    filepath = formatstr.format(embedtext=embedding, model=modelname)
    try:
        with open(config["selectvec"], "r") as f:
            selectvec = json.load(f)
    except FileNotFoundError:
        print("Error: {config['selectvec']} not found.")

    # read in precomputed, but retain only the embeddings for the analysis
    try:
        if os.path.exists(filepath):  # type: ignore
            data_array = np.loadtxt(filepath, delimiter=",")  # type: ignore
            data_array = data_array[selectvec,]
            return data_array
        # else:
        #     return None
    except FileNotFoundError:
        # Handle the case where the file does not exist
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except PermissionError:
        # Handle the case where there are insufficient permissions to access the file
        print(f"Error: Permission denied for '{filepath}'.")
        return None
    except IOError as e:
        # Handle other general input/output errors
        print(f"An I/O error occurred: {e}")
        return None
    except Exception as e:
        # Catch any other unexpected exceptions
        print(f"An unexpected error occurred: {e}")
        return None


def _is_float_string(s) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_configs(path: str) -> Dict:
    config = configparser.ConfigParser()
    try:
        config.read(path)
        config_dict = config_to_dict(config)
        return config_dict

    except FileNotFoundError:
        # Handle the case where the file does not exist
        print(f"Error: The file '{path}' was not found.")
        return {}
    except PermissionError:
        # Handle the case where there are insufficient permissions to access the file
        print(f"Error: Permission denied for '{path}'.")
        return {}
    except IOError as e:
        # Handle other general input/output errors
        print(f"An I/O error occurred: {e}")
        return {}
    except Exception as e:
        # Catch any other unexpected exceptions
        print(f"An unexpected error occurred: {e}")
        return {}


def config_to_dict(configuration) -> Dict:
    config_dict = {}
    for section in configuration.sections():
        config_dict[section] = {}
        for option in configuration.options(section):

            value = configuration.get(section, option).lstrip()

            if "\n" in value:
                # print(f"Match for section: {section}, option: {option}, value: {value}\n")
                config_dict[section][option] = [
                    item.strip() for item in value.splitlines() if item.strip()
                ]
            elif value.lower() in ("true", "false"):
                config_dict[section][option] = configuration.getboolean(section, option)
            elif value.isdigit():
                config_dict[section][option] = configuration.getint(section, option)
            elif value.lower() in ("inf", "infinity", "-inf"):
                value = float(value)
            elif _is_float_string(value):
                # print(f"Match for section: {section}, option: {option}, value: {value}\n")
                config_dict[section][option] = configuration.getfloat(section, option)
            elif "(" in value:
                value = ast.literal_eval(value)
                config_dict[section][option] = value
            elif "None" in value:
                value = None
                config_dict[section][option] = value
            else:
                config_dict[section][option] = value

    return config_dict


logger = setup_logging()


def auto_cast(val):
    if val is None or val == "":
        return None
    if isinstance(val, str):
        v = val.strip()

        # --- Try literal_eval for tuple/list/dict/None ---
        if (v.startswith("(") and v.endswith(")")) or (
            v.startswith("[") and v.endswith("]")
        ):
            try:
                return ast.literal_eval(v)
            except Exception:
                pass  # fall back if it isn’t valid Python literal

        # --- booleans ---
        if v.lower() in ("true", "false"):
            return v.lower() == "true"

        # --- infinity ---
        if v.lower() in ("inf", "infinity", "-inf"):
            return float(v)

        # --- numbers ---
        try:
            return int(v)
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            pass

        # --- fallback string ---
        return v
    return val
