"""
load_dataset.py
---------------
Utilities for loading and validating the INFACT TSV dataset.

The INFACT dataset is a tab-separated file with the following columns:
    record_id, source_url, date_verified, author_claim, source_outlet,
    claim_text, context, verification_scope, verification, conclusion,
    domain_claim, verdict_original

Example usage
-------------
    from src.data.load_dataset import load_infact, validate_dataset

    df = load_infact("data/infact_dataset.tsv")
    validate_dataset(df)
    print(df.head())
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "record_id",
    "source_url",
    "date_verified",
    "author_claim",
    "source_outlet",
    "claim_text",
    "context",
    "verification_scope",
    "verification",
    "conclusion",
    "domain_claim",
    "verdict_original",
]


def load_infact(
    path: str | Path,
    encoding: str = "utf-8",
    sep: str = "\t",
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """Load the INFACT dataset from a TSV file.

    Parameters
    ----------
    path:
        Path to the TSV dataset file.
    encoding:
        File encoding (default: ``"utf-8"``).
    sep:
        Column separator (default: tab).
    drop_duplicates:
        Whether to drop exact duplicate rows (default: ``True``).

    Returns
    -------
    pd.DataFrame
        DataFrame with all dataset columns.  The ``date_verified`` column is
        parsed as ``datetime64``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path, sep=sep, encoding=encoding, low_memory=False)
    logger.info("Loaded %d rows from %s", len(df), path)

    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed:
            logger.info("Dropped %d duplicate rows.", removed)

    # Parse date column if present; the INFACT dataset uses DD.MM.YYYY format
    if "date_verified" in df.columns:
        df["date_verified"] = pd.to_datetime(
            df["date_verified"], format="%d.%m.%Y", errors="coerce"
        )

    return df


def validate_dataset(df: pd.DataFrame, extra_columns: Optional[list[str]] = None) -> None:
    """Validate that a DataFrame contains all required INFACT columns.

    Parameters
    ----------
    df:
        DataFrame to validate.
    extra_columns:
        Additional column names to check beyond the standard set.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    expected = list(REQUIRED_COLUMNS)
    if extra_columns:
        expected += extra_columns

    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    logger.info("Dataset validation passed. Shape: %s", df.shape)


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Return a dictionary with basic dataset statistics.

    Parameters
    ----------
    df:
        Loaded INFACT DataFrame.

    Returns
    -------
    dict
        Keys include ``n_records``, ``n_columns``, ``verdict_counts``,
        ``domain_counts``, ``missing_per_column``, and ``date_range``.
    """
    summary: dict = {
        "n_records": len(df),
        "n_columns": len(df.columns),
        "verdict_counts": df["verdict_original"].value_counts().to_dict()
        if "verdict_original" in df.columns
        else {},
        "domain_counts": df["domain_claim"].value_counts().to_dict()
        if "domain_claim" in df.columns
        else {},
        "missing_per_column": df.isnull().sum().to_dict(),
    }

    if "date_verified" in df.columns:
        valid_dates = df["date_verified"].dropna()
        summary["date_range"] = {
            "min": str(valid_dates.min()) if len(valid_dates) else None,
            "max": str(valid_dates.max()) if len(valid_dates) else None,
        }

    return summary


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO)
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/infact_dataset.tsv"

    dataset = load_infact(data_path)
    validate_dataset(dataset)
    summary = get_dataset_summary(dataset)
    print(json.dumps(summary, indent=2, default=str))
