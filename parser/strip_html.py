import json
import argparse
import logging
import unicodedata
import yaml  # pip install pyyaml
import csv
import warnings
from pathlib import Path
from pydantic import BaseModel
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning  # type: ignore
from typing import Any, Type, List, Optional, Dict, Union
from CDE_Schema import CDEForm, CDEItem

# from strip_html import strip_html


# === MODEL REGISTRY ===
MODEL_REGISTRY: dict[str, Type[BaseModel]] = {
    "CDE": CDEItem,
    "Form": CDEForm,
}


# === LOGGING ===
def configure_logging(verbosity: int, logfile: Union[str, None]):
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    handlers = [logging.StreamHandler()]
    if logfile:
        handlers = [logging.FileHandler(logfile, mode="a", encoding="utf-8")]
        # handlers.append(logfile)

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=handlers,
    )


# === IO ===
def load_json(filepath: Path) -> Union[list, dict]:
    with filepath.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data: Any, output_path: Path, fmt: str, pretty: bool):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2 if pretty else None, ensure_ascii=False)

    elif fmt == "yaml":
        with output_path.open("w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    elif fmt == "csv":
        if isinstance(data, list) and all(isinstance(row, dict) for row in data):
            fieldnames = sorted(set().union(*(row.keys() for row in data)))
            with output_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
        else:
            raise ValueError("CSV format only supports list of dicts.")
    else:
        raise ValueError(f"Unsupported format: {fmt}")


# === CLEANING ===
def normalize_string(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip().lower()


def strip_html(text: str) -> str:
    if text is None:
        return None
    #    print(f"stripping html: {text}")
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def clean_text_values(obj: Any, set_keys) -> Any:
    if isinstance(obj, str):
        return strip_html(obj)
    elif isinstance(obj, BaseModel):
        cleaned = {
            k: clean_text_values(v, set_keys)
            for k, v in obj.model_dump(
                exclude_unset=True if set_keys else False,
                exclude_none=True if set_keys else False,
            ).items()
        }
        return obj.__class__(**cleaned)
    elif isinstance(obj, list):
        return [clean_text_values(item, set_keys) for item in obj]
    elif isinstance(obj, dict):
        return {k: clean_text_values(v, set_keys) for k, v in obj.items()}
    else:
        return obj


# === CORE ===
def process_data(
    data: Union[list, dict], model_class: Type[BaseModel], set_keys
) -> list[dict]:
    logging.debug(f"Raw input type: {type(data).__name__}")
    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        raise ValueError("Input must be a dict or list of dicts.")

    models = [model_class(**item) for item in data]
    cleaned = [clean_text_values(model, set_keys) for model in models]
    return [model.model_dump(by_alias=True) for model in cleaned]


def process_file(
    filepath: Path,
    outdir: Path,
    model_class: Type[BaseModel],
    fmt: str,
    dry_run: bool,
    set_keys: bool,
    pretty: bool,
):
    logging.info(f"Processing: {filepath}")
    try:
        raw_data = load_json(filepath)
        cleaned_data = process_data(raw_data, model_class, set_keys)

        output_path = outdir / f"{filepath.stem}_nohtml.{fmt}"

        if dry_run:
            logging.info(f"[Dry-run] Would write: {output_path}")
        else:
            save_data(cleaned_data, output_path, fmt, pretty)
            logging.info(f"Saved cleaned data to: {output_path}")

    except Exception as e:
        logging.error(f"Error processing {filepath.name}: {e}")


# === CLI ===
def main():
    parser = argparse.ArgumentParser(
        description="Clean and normalize string fields in structured JSON via Pydantic models."
    )
    parser.add_argument("filenames", nargs="+", help="Input JSON file(s)")
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        choices=MODEL_REGISTRY.keys(),
        help="Model to use for validation",
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Directory for output files (default: current directory)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "yaml", "csv"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not write output files"
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        action="count",
        default=1,
        help="Increase verbosity level (-vv for debug)",
    )
    parser.add_argument("--logfile", help="Optional log file path")
    parser.add_argument(
        "--pretty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Produce pretty (default: --pretty) or minified (--no-pretty) JSON (no whitespace)",
    )
    parser.add_argument(
        "--set-keys",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save model with keys only represented if they are set (no null, None, or empty sets)",
    )

    args = parser.parse_args()
    print(args)
    configure_logging(args.verbosity, args.logfile)

    model_class = MODEL_REGISTRY[args.model]
    outdir = Path(args.outdir)

    for filename in args.filenames:
        filepath = Path(filename)
        if not filepath.is_file():
            logging.warning(f"Skipping: {filename} is not a valid file.")
            continue
        process_file(
            filepath,
            outdir,
            model_class,
            args.format,
            args.dry_run,
            args.set_keys,
            args.pretty,
        )


if __name__ == "__main__":
    main()
