"""
linguistic_bias.py
------------------
Linguistic framing analysis for the INFACT corpus.

Examines three types of discourse markers that signal speaker stance and
epistemic commitment:

* **Hedges** — words that weaken a claim (e.g., "poate", "probabil",
  "approximately")
* **Certainty markers** — words that strengthen a claim (e.g., "absolut",
  "clar", "definitely")
* **Authority markers** — references to institutional or expert sources that
  lend credibility (e.g., "conform", "potrivit", "according to")

For each category, the module counts occurrences per claim, computes
distribution statistics, and compares marker frequencies across verdict
categories.

Example usage
-------------
    from src.data.load_dataset import load_infact
    from src.data.label_mapping import apply_label_mapping
    from src.analysis.linguistic_bias import run_linguistic_analysis

    df = load_infact("data/infact_dataset.tsv")
    df = apply_label_mapping(df)
    results = run_linguistic_analysis(df, output_dir="results/reports")
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FIGURE_DPI = 150

# ---------------------------------------------------------------------------
# Marker lexicons (Romanian + English fallbacks)
# ---------------------------------------------------------------------------

HEDGE_MARKERS: list[str] = [
    # Romanian
    "poate", "probabil", "posibil", "aproximativ", "în jur de", "cam",
    "pare", "se pare", "presupun", "presupunem", "aparent", "oarecum",
    "relativ", "destul de", "cumva",
    # English
    "maybe", "perhaps", "possibly", "probably", "approximately", "roughly",
    "seems", "appear", "suggest", "indicate", "likely", "somewhat",
]

CERTAINTY_MARKERS: list[str] = [
    # Romanian
    "absolut", "sigur", "cert", "clar", "evident", "fără îndoială",
    "indubitabil", "neîndoielnic", "categoric", "cu certitudine", "în mod clar",
    # English
    "absolutely", "certainly", "clearly", "definitely", "undoubtedly",
    "obviously", "without doubt", "surely", "indeed",
]

AUTHORITY_MARKERS: list[str] = [
    # Romanian
    "conform", "potrivit", "după", "declară", "afirmă", "susține",
    "potrivit declarațiilor", "oficialii", "experții", "studiul",
    "raportul", "institutul", "ministerul", "guvernul", "parlamentul",
    # English
    "according to", "as stated by", "experts say", "officials", "report",
    "study", "institute", "government", "parliament", "ministry",
]


# ---------------------------------------------------------------------------
# Counting helpers
# ---------------------------------------------------------------------------

def count_markers(text: str, markers: list[str]) -> int:
    """Count occurrences of any marker pattern in *text*.

    The match is case-insensitive and uses whole-token boundaries.

    Parameters
    ----------
    text:
        Input string to search.
    markers:
        List of marker strings (may include spaces for multi-word phrases).

    Returns
    -------
    int
        Total count of all marker occurrences.
    """
    if not isinstance(text, str) or not text.strip():
        return 0
    text_lower = text.lower()
    total = 0
    for marker in markers:
        # Use word-boundary matching for single-word markers
        if " " in marker:
            total += text_lower.count(marker.lower())
        else:
            total += len(re.findall(rf"\b{re.escape(marker.lower())}\b", text_lower))
    return total


def add_marker_counts(
    df: pd.DataFrame,
    text_col: str = "claim_text",
) -> pd.DataFrame:
    """Append hedge, certainty, and authority count columns to *df*.

    Parameters
    ----------
    df:
        INFACT DataFrame.
    text_col:
        Column to analyse.

    Returns
    -------
    pd.DataFrame
        Copy with three new columns:
        ``hedge_count``, ``certainty_count``, ``authority_count``.
    """
    df = df.copy()
    texts = df[text_col].fillna("")
    df["hedge_count"] = texts.apply(lambda t: count_markers(t, HEDGE_MARKERS))
    df["certainty_count"] = texts.apply(lambda t: count_markers(t, CERTAINTY_MARKERS))
    df["authority_count"] = texts.apply(lambda t: count_markers(t, AUTHORITY_MARKERS))
    return df


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def compute_marker_statistics(df: pd.DataFrame) -> dict:
    """Compute summary statistics for each marker category.

    Parameters
    ----------
    df:
        DataFrame with ``hedge_count``, ``certainty_count``,
        ``authority_count`` columns (added by :func:`add_marker_counts`).

    Returns
    -------
    dict
        Mean, median, std, and max per marker category.
    """
    stats: dict = {}
    for col in ("hedge_count", "certainty_count", "authority_count"):
        if col not in df.columns:
            continue
        stats[col] = {
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std()),
            "max": int(df[col].max()),
            "pct_nonzero": float((df[col] > 0).mean()),
        }
    return stats


def compare_markers_by_verdict(
    df: pd.DataFrame,
    label_col: str = "verdict_normalized",
) -> pd.DataFrame:
    """Compare average marker counts across verdict categories.

    Parameters
    ----------
    df:
        DataFrame with marker count and label columns.
    label_col:
        Column containing verdict labels.

    Returns
    -------
    pd.DataFrame
        Pivot table: rows = verdict, columns = marker type, values = mean count.
    """
    marker_cols = [c for c in ("hedge_count", "certainty_count", "authority_count") if c in df.columns]
    if not marker_cols or label_col not in df.columns:
        return pd.DataFrame()
    return df.groupby(label_col)[marker_cols].mean().round(4)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_marker_by_verdict(
    df: pd.DataFrame,
    label_col: str = "verdict_normalized",
    output_dir: str = "results/figures",
    show: bool = False,
) -> Optional[Path]:
    """Plot mean marker counts per verdict as a grouped bar chart.

    Parameters
    ----------
    df:
        INFACT DataFrame with marker count columns.
    label_col:
        Column containing verdict labels.
    output_dir:
        Directory where the figure is saved.
    show:
        Whether to call ``plt.show()``.

    Returns
    -------
    Path | None
    """
    pivot = compare_markers_by_verdict(df, label_col=label_col)
    if pivot.empty:
        logger.warning("No data for marker-by-verdict plot — skipping.")
        return None

    ax = pivot.plot(kind="bar", figsize=(12, 5), edgecolor="white")
    ax.set_title("Mean Linguistic Marker Counts per Verdict", fontsize=14)
    ax.set_xlabel("Verdict", fontsize=12)
    ax.set_ylabel("Mean Count", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Marker Type")
    plt.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "marker_counts_by_verdict.png"
    ax.get_figure().savefig(save_path, dpi=FIGURE_DPI)
    logger.info("Saved figure: %s", save_path)
    if show:
        plt.show()
    plt.close(ax.get_figure())
    return save_path


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_linguistic_analysis(
    df: pd.DataFrame,
    text_col: str = "claim_text",
    output_dir: str = "results/reports",
    figure_dir: str = "results/figures",
    show: bool = False,
) -> dict:
    """Run the complete linguistic framing analysis.

    Parameters
    ----------
    df:
        INFACT DataFrame (label mapping recommended).
    text_col:
        Column to analyse for markers.
    output_dir:
        Directory for JSON report.
    figure_dir:
        Directory for figures.
    show:
        Whether to display plots interactively.

    Returns
    -------
    dict
        Dictionary with global statistics and per-verdict breakdown.
    """
    df = add_marker_counts(df, text_col=text_col)
    stats = compute_marker_statistics(df)
    pivot = compare_markers_by_verdict(df)
    verdict_breakdown = pivot.to_dict() if not pivot.empty else {}

    report = {
        "global_statistics": stats,
        "verdict_breakdown": verdict_breakdown,
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "linguistic_framing_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Saved linguistic framing report to %s", report_path)

    plot_marker_by_verdict(df, output_dir=figure_dir, show=show)

    print("\n--- Linguistic Framing Summary ---")
    for cat, s in stats.items():
        print(f"  {cat:<20}  mean={s['mean']:.3f}  pct_nonzero={s['pct_nonzero']:.2%}")
    print()

    return report


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    from src.data_preprocessing.label_mapping import apply_label_mapping
    from src.data_preprocessing.load_dataset import load_infact, validate_dataset

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/infact_dataset.tsv"
    dataset = load_infact(data_path)
    validate_dataset(dataset)
    dataset = apply_label_mapping(dataset)
    run_linguistic_analysis(dataset)
