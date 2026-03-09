"""
io.py
-----
I/O utility functions for the INFACT pipeline.

Provides helpers for:
- Saving and loading JSON files
- Saving DataFrames to CSV / LaTeX
- Ensuring output directories exist

Example usage
-------------
    from src.utils.io import save_json, save_dataframe, ensure_dir

    ensure_dir("results/tables")
    save_json({"accuracy": 0.87}, "results/reports/metrics.json")
    save_dataframe(df, "results/tables/results.csv")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def ensure_dir(path: str | Path) -> Path:
    """Create *path* and any intermediate directories if they do not exist.

    Parameters
    ----------
    path:
        Directory path to create.

    Returns
    -------
    Path
        The resolved directory path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """Serialise *data* to a JSON file.

    Parameters
    ----------
    data:
        JSON-serialisable Python object.
    path:
        Destination file path.
    indent:
        JSON indentation level.

    Returns
    -------
    Path
        Path to the saved file.
    """
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, default=str)
    logger.info("Saved JSON: %s", path)
    return path


def load_json(path: str | Path) -> Any:
    """Load and return a JSON file.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    Any
        Deserialised Python object.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_dataframe(
    df: pd.DataFrame,
    path: str | Path,
    index: bool = True,
    fmt: str = "csv",
) -> Path:
    """Save a DataFrame to CSV or LaTeX format.

    Parameters
    ----------
    df:
        DataFrame to save.
    path:
        Destination file path (the extension is used if ``fmt`` is not given
        explicitly).
    index:
        Whether to write the DataFrame index.
    fmt:
        ``"csv"`` or ``"latex"``.

    Returns
    -------
    Path
        Path to the saved file.

    Raises
    ------
    ValueError
        If *fmt* is not recognised.
    """
    path = Path(path)
    ensure_dir(path.parent)

    if fmt == "csv":
        df.to_csv(path, index=index)
    elif fmt == "latex":
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(df.to_latex(index=index))
    else:
        raise ValueError(f"Unsupported format '{fmt}'. Choose 'csv' or 'latex'.")

    logger.info("Saved DataFrame (%d rows) to %s", len(df), path)
    return path


if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)
    temp_dir = Path(tempfile.gettempdir()) / "infact_test"
    ensure_dir(temp_dir)
    save_json({"test": True, "value": 42}, temp_dir / "test.json")
    loaded = load_json(temp_dir / "test.json")
    assert loaded["value"] == 42
    print("io.py smoke test passed.")
