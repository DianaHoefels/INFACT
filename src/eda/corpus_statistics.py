"""
corpus_statistics.py
--------------------
Exploratory data analysis (EDA) for the INFACT corpus.

Provides functions to compute and visualise:
- Overall dataset statistics (record counts, missing values, date range)
- Verdict label distribution (pie chart + bar chart)
- Domain distribution
- Temporal distribution of claims
- Claim and context text length distributions
- Author / outlet frequency analysis

All plotting functions save figures to ``results/figures/`` by default.

Example usage
-------------
    from src.data.load_dataset import load_infact
    from src.data.label_mapping import apply_label_mapping
    from src.eda.corpus_statistics import run_eda

    df = load_infact("data/infact_dataset.tsv")
    df = apply_label_mapping(df)
    run_eda(df, output_dir="results/figures")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FIGURE_DPI = 150


# Optional: Romanian-to-English mapping for domains (extend as needed)
DOMAIN_TRANSLATION = {
    # Example mappings; extend as needed based on dataset values
    "politică": "Politics",
    "sănătate": "Health",
    "educație": "Education",
    "economie": "Economy",
    "justiție": "Justice",
    "mediu": "Environment",
    # Add more as needed
}

def translate_domain(domain: str) -> str:
    """Translate Romanian domain names to English for plotting."""
    return DOMAIN_TRANSLATION.get(domain, domain)


# ---------------------------------------------------------------------------
# Text-length helpers
# ---------------------------------------------------------------------------

def add_text_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """Append word-count columns for ``claim_text`` and ``context``.

    Parameters
    ----------
    df:
        INFACT DataFrame.

    Returns
    -------
    pd.DataFrame
        Copy with added ``claim_len`` and ``context_len`` columns.
    """
    df = df.copy()
    if "claim_text" in df.columns:
        df["claim_len"] = df["claim_text"].fillna("").apply(lambda x: len(str(x).split()))
    if "context" in df.columns:
        df["context_len"] = df["context"].fillna("").apply(lambda x: len(str(x).split()))
    return df


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def print_basic_stats(df: pd.DataFrame) -> None:
    """Print basic corpus statistics to stdout.

    Parameters
    ----------
    df:
        INFACT DataFrame (may include normalised label columns).
    """
    print("=" * 60)
    print("INFACT Corpus — Basic Statistics")
    print("=" * 60)
    print(f"  Total records   : {len(df):,}")
    print(f"  Total columns   : {len(df.columns)}")
    if "date_verified" in df.columns:
        valid = df["date_verified"].dropna()
        if len(valid):
            print(f"  Date range      : {valid.min().date()} → {valid.max().date()}")
    print("\nMissing values per column:")
    missing = df.isnull().sum()
    for col, cnt in missing[missing > 0].items():
        pct = 100.0 * cnt / len(df)
        print(f"  {col:<28} {cnt:>5} ({pct:.1f}%)")
    if missing.sum() == 0:
        print("  None")
    print()


def print_verdict_distribution(df: pd.DataFrame, label_col: str = "verdict_original") -> None:
    """Print the verdict label distribution.

    Parameters
    ----------
    df:
        INFACT DataFrame.
    label_col:
        Column containing verdict labels.
    """
    if label_col not in df.columns:
        logger.warning("Column '%s' not found.", label_col)
        return
    counts = df[label_col].value_counts()
    print(f"Verdict distribution ({label_col}):")
    for label, count in counts.items():
        pct = 100.0 * count / len(df)
        print(f"  {label:<30} {count:>5}  ({pct:.1f}%)")
    print()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_verdict_distribution(
    df: pd.DataFrame,
    label_col: str = "verdict_original",
    output_dir: str = "results/figures",
    show: bool = False,
) -> Optional[Path]:
    """Plot and save a bar chart of verdict label counts.

    Parameters
    ----------
    df:
        INFACT DataFrame.
    label_col:
        Column containing verdict labels.
    output_dir:
        Directory where the figure is saved.
    show:
        Whether to call ``plt.show()`` interactively.

    Returns
    -------
    Path | None
        Path to the saved figure, or ``None`` if the column is missing.
    """
    if label_col not in df.columns:
        logger.warning("Column '%s' not found — skipping plot.", label_col)
        return None

    counts = df[label_col].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    # Use English for all labels
    ax.set_title(f"Verdict Distribution — {label_col.replace('verdict_original', 'Original Verdict').replace('verdict_normalized', 'Normalized Verdict')}", fontsize=14)
    ax.set_xlabel("Verdict", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / f"verdict_distribution_{label_col}.png"
    fig.savefig(save_path, dpi=FIGURE_DPI)
    logger.info("Saved figure: %s", save_path)
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def plot_domain_distribution(
    df: pd.DataFrame,
    output_dir: str = "results/figures",
    show: bool = False,
) -> Optional[Path]:
    """Plot and save a bar chart of domain counts.

    Parameters
    ----------
    df:
        INFACT DataFrame.
    output_dir:
        Directory where the figure is saved.
    show:
        Whether to call ``plt.show()`` interactively.

    Returns
    -------
    Path | None
        Path to the saved figure, or ``None`` if the column is missing.
    """
    if "domain_claim" not in df.columns:
        logger.warning("Column 'domain_claim' not found — skipping plot.")
        return None

    counts = df["domain_claim"].value_counts().head(20)
    # Translate domain names for plotting
    counts.index = [translate_domain(domain) for domain in counts.index]
    fig, ax = plt.subplots(figsize=(12, 5))
    counts.plot(kind="bar", ax=ax, color="teal", edgecolor="white")
    ax.set_title("Top 20 Claim Domains", fontsize=14)
    ax.set_xlabel("Domain", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "domain_distribution.png"
    fig.savefig(save_path, dpi=FIGURE_DPI)
    logger.info("Saved figure: %s", save_path)
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def plot_temporal_distribution(
    df: pd.DataFrame,
    output_dir: str = "results/figures",
    show: bool = False,
) -> Optional[Path]:
    """Plot monthly claim counts over time.

    Parameters
    ----------
    df:
        INFACT DataFrame with a ``date_verified`` datetime column.
    output_dir:
        Directory where the figure is saved.
    show:
        Whether to call ``plt.show()`` interactively.

    Returns
    -------
    Path | None
        Path to the saved figure, or ``None`` if the column is missing.
    """
    if "date_verified" not in df.columns:
        logger.warning("Column 'date_verified' not found — skipping temporal plot.")
        return None

    monthly = df.set_index("date_verified").resample("ME").size()
    if monthly.empty:
        logger.warning("No valid dates found — skipping temporal plot.")
        return None

    fig, ax = plt.subplots(figsize=(14, 4))
    monthly.plot(ax=ax, color="darkorange", linewidth=1.5)
    ax.fill_between(monthly.index, monthly.values, alpha=0.3, color="darkorange")
    ax.set_title("Monthly Claim Counts Over Time", fontsize=14)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Number of Claims", fontsize=12)
    plt.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "temporal_distribution.png"
    fig.savefig(save_path, dpi=FIGURE_DPI)
    logger.info("Saved figure: %s", save_path)
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def plot_text_length_distribution(
    df: pd.DataFrame,
    output_dir: str = "results/figures",
    show: bool = False,
) -> Optional[Path]:
    """Plot word-count histograms for claim and context columns.

    Parameters
    ----------
    df:
        INFACT DataFrame (``claim_len`` / ``context_len`` added automatically
        if not already present).
    output_dir:
        Directory where the figure is saved.
    show:
        Whether to call ``plt.show()`` interactively.

    Returns
    -------
    Path | None
        Path to the saved figure.
    """
    df = add_text_lengths(df)
    cols = [c for c in ("claim_len", "context_len") if c in df.columns]
    if not cols:
        logger.warning("No text length columns available — skipping plot.")
        return None

    fig, axes = plt.subplots(1, len(cols), figsize=(7 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        # Use English for column titles
        title = "Claim Length Distribution" if col == "claim_len" else "Context Length Distribution"
        ax.hist(df[col], bins=40, color="mediumslateblue", edgecolor="white")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Word count", fontsize=11)
        ax.set_ylabel("Frequency", fontsize=11)

    plt.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "text_length_distribution.png"
    fig.savefig(save_path, dpi=FIGURE_DPI)
    logger.info("Saved figure: %s", save_path)
    if show:
        plt.show()
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_eda(
    df: pd.DataFrame,
    output_dir: str = "results/figures",
    show: bool = False,
) -> None:
    """Run the complete EDA suite and save all figures.

    Parameters
    ----------
    df:
        INFACT DataFrame (label mapping already applied is recommended).
    output_dir:
        Directory where all figures are saved.
    show:
        Whether to display figures interactively.
    """
    print_basic_stats(df)

    for col in ("verdict_original", "verdict_normalized"):
        if col in df.columns:
            print_verdict_distribution(df, label_col=col)
            plot_verdict_distribution(df, label_col=col, output_dir=output_dir, show=show)

    plot_domain_distribution(df, output_dir=output_dir, show=show)
    plot_temporal_distribution(df, output_dir=output_dir, show=show)
    plot_text_length_distribution(df, output_dir=output_dir, show=show)

    logger.info("EDA complete. Figures saved to: %s", output_dir)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    from src.data.label_mapping import apply_label_mapping
    from src.data.load_dataset import load_infact, validate_dataset

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/infact_dataset.tsv"
    dataset = load_infact(data_path)
    validate_dataset(dataset)
    dataset = apply_label_mapping(dataset)
    run_eda(dataset)
