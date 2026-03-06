"""Verdict label engineering for the RoFACT dataset."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.utils.io import save_markdown, save_table, setup_logging
from src.utils.text import normalize_diacritics

logger = setup_logging(__name__)

# ---------------------------------------------------------------------------
# Epistemic mapping – keys use new-style comma-below diacritics
# ---------------------------------------------------------------------------

EPISTEMIC_MAPPING: dict[str, str] = {
    # False
    "fals": "false",
    "fals parțial": "false",
    # True
    "adevărat": "true",
    "adevarat": "true",
    # Partial
    "trunchiat": "partial",
    "parțial adevărat": "partial",
    "partial adevarat": "partial",
    "parțial fals": "partial",
    "partial fals": "partial",
    "în mare parte adevărat": "partial",
    "în mare parte fals": "partial",
    "in mare parte adevarat": "partial",
    "in mare parte fals": "partial",
    "parțial": "partial",
    "partial": "partial",
    # Unverifiable
    "imposibil de verificat": "unverifiable",
    "nu se poate verifica": "unverifiable",
    "neverificabil": "unverifiable",
    # Other
    "inexplicabil": "other",
    "numai cu sprijin instituțional": "other",
    "numai cu sprijin institutional": "other",
    "nedovedit": "other",
    "în curs de verificare": "other",
    "in curs de verificare": "other",
}

BINARY_EXCLUDED: set[str] = {"unverifiable", "other"}


# ---------------------------------------------------------------------------
# Mapping functions
# ---------------------------------------------------------------------------


def map_epistemic(verdict) -> str:
    """Normalise diacritics, lowercase, and look up in EPISTEMIC_MAPPING.

    Returns "unknown" when no match is found.
    """
    if not isinstance(verdict, str) or not verdict.strip():
        return "unknown"
    normalised = normalize_diacritics(verdict).strip().lower()
    return EPISTEMIC_MAPPING.get(normalised, "unknown")


def map_binary(epistemic: str) -> str | None:
    """Map epistemic label to binary classification target.

    Returns:
        "positive"  – if epistemic == "true"
        "negative"  – if epistemic in {"false", "partial"}
        None        – if epistemic in BINARY_EXCLUDED or "unknown"
    """
    if epistemic == "true":
        return "positive"
    if epistemic in {"false", "partial"}:
        return "negative"
    return None


def apply_label_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Add *epistemic_outcome* and *verification_binary* columns."""
    df = df.copy()
    df["epistemic_outcome"] = df["verdict_original"].apply(map_epistemic)
    df["verification_binary"] = df["epistemic_outcome"].apply(map_binary)
    return df


def label_mapping_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarising epistemic and binary label counts."""
    epistemic_counts = (
        df["epistemic_outcome"]
        .value_counts()
        .rename_axis("epistemic_outcome")
        .reset_index(name="count")
    )
    epistemic_counts["percent"] = (
        epistemic_counts["count"] / epistemic_counts["count"].sum() * 100
    ).round(2)

    binary_counts = (
        df["verification_binary"]
        .value_counts(dropna=False)
        .rename_axis("verification_binary")
        .reset_index(name="binary_count")
    )
    binary_counts["binary_percent"] = (
        binary_counts["binary_count"] / len(df) * 100
    ).round(2)

    summary = epistemic_counts.merge(
        binary_counts.rename(columns={"verification_binary": "epistemic_outcome"}),
        on="epistemic_outcome",
        how="left",
    )
    return summary


def run_label_engineering(
    df: pd.DataFrame,
    output_dir: str = "results",
) -> pd.DataFrame:
    """Apply label engineering, save summary, and return enriched DataFrame.

    Saves:
      - results/tables/label_mapping_summary.csv
    """
    df = apply_label_engineering(df)

    summary = label_mapping_summary(df)
    tables_dir = Path(output_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    save_table(summary, tables_dir / "label_mapping_summary.csv")

    logger.info(
        "Label engineering done. Epistemic distribution:\n%s",
        df["epistemic_outcome"].value_counts().to_string(),
    )
    return df
