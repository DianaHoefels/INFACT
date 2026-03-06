"""Load, validate, clean, and pre-process the RoFACT dataset."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import save_markdown, save_tsv, setup_logging
from src.utils.text import normalize_diacritics, word_count

logger = setup_logging(__name__)

REQUIRED_COLUMNS: list[str] = [
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

TEXT_FIELDS: list[str] = [
    "claim_text",
    "context",
    "verification_scope",
    "verification",
    "conclusion",
]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_tsv(path: str | os.PathLike) -> pd.DataFrame:
    """Load a UTF-8 TSV file; text columns default NaN → empty string."""
    df = pd.read_csv(
        path,
        sep="\t",
        encoding="utf-8",
        dtype=str,
        keep_default_na=False,
        na_values=[""],
    )
    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_columns(df: pd.DataFrame) -> dict:
    """Return a report dict describing missing columns and basic shape info."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    report: dict = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "missing_required_columns": missing,
        "extra_columns": extra,
        "all_required_present": len(missing) == 0,
    }
    if missing:
        logger.warning("Missing required columns: %s", missing)
    return report


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows with duplicate record_id values."""
    if "record_id" not in df.columns:
        return pd.DataFrame()
    duplicated_mask = df.duplicated(subset=["record_id"], keep=False)
    return df[duplicated_mask].copy()


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date_verified (DD.MM.YYYY) and add year_verified column."""
    df = df.copy()
    if "date_verified" in df.columns:
        df["date_verified"] = pd.to_datetime(
            df["date_verified"], format="%d.%m.%Y", errors="coerce"
        )
        df["year_verified"] = df["date_verified"].dt.year.astype("Int64")
    return df


def compute_section_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """Add word_count_{field} columns for each field in TEXT_FIELDS."""
    df = df.copy()
    for field in TEXT_FIELDS:
        if field in df.columns:
            df[f"word_count_{field}"] = df[field].apply(word_count)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from all string columns."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(
            lambda x: x.strip() if isinstance(x, str) else x
        )
    # Fill NaN in TEXT_FIELDS with empty string
    for field in TEXT_FIELDS:
        if field in df.columns:
            df[field] = df[field].fillna("")
    return df


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_validation(
    path: str | os.PathLike,
    output_dir: str = "results",
) -> tuple[pd.DataFrame, dict]:
    """Run the full load → validate → clean → parse → length pipeline.

    Saves:
      - results/tables/rofact_cleaned.tsv
      - results/reports/data_validation_report.md

    Returns (cleaned_df, report_dict).
    """
    logger.info("Loading dataset from %s", path)
    df = load_tsv(path)

    report = validate_columns(df)
    dups = check_duplicates(df)
    report["duplicate_record_ids"] = int(len(dups))

    # Missingness per column
    missingness: dict[str, int] = {}
    for col in df.columns:
        n_missing = int(df[col].isna().sum()) + int((df[col] == "").sum())
        missingness[col] = n_missing
    report["missingness"] = missingness

    df = clean_dataframe(df)
    df = parse_dates(df)
    df = compute_section_lengths(df)

    # Save outputs
    tables_dir = Path(output_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    save_tsv(df, tables_dir / "rofact_cleaned.tsv")

    # Build markdown report
    lines = [
        "# Data Validation Report\n",
        f"**Rows:** {report['total_rows']}  \n"
        f"**Columns:** {report['total_columns']}\n\n",
        "## Required Columns\n",
    ]
    if report["all_required_present"]:
        lines.append("✅ All required columns are present.\n\n")
    else:
        lines.append(
            f"❌ Missing columns: {report['missing_required_columns']}\n\n"
        )
    if report["extra_columns"]:
        lines.append(f"**Extra columns:** {report['extra_columns']}\n\n")

    lines.append(f"## Duplicates\n{report['duplicate_record_ids']} duplicate record_id(s).\n\n")

    lines.append("## Missingness per Column\n| Column | Missing |\n|---|---|\n")
    for col, n in missingness.items():
        lines.append(f"| {col} | {n} |\n")

    wc_cols = [c for c in df.columns if c.startswith("word_count_")]
    if wc_cols:
        lines.append("\n## Section Length Statistics (word count)\n")
        stats = df[wc_cols].describe().round(1)
        lines.append(stats.to_markdown() + "\n")

    md_text = "".join(lines)
    reports_dir = Path(output_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    save_markdown(md_text, reports_dir / "data_validation_report.md")

    logger.info(
        "Validation complete: %d rows, %d duplicates",
        report["total_rows"],
        report["duplicate_record_ids"],
    )
    return df, report
