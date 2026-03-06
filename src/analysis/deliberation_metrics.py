"""
deliberation_metrics.py
-----------------------
Deliberation-aware analysis of discourse representation in the INFACT corpus.

This module operationalises deliberative quality indicators drawn from
political communication and deliberation theory, including:

- **Voice diversity** — how many distinct authors/outlets are represented
- **Domain coverage** — breadth of political domains covered
- **Temporal spread** — distribution of claims over time
- **Verification scope distribution** — how often national vs. local claims
  are checked
- **Conclusion sentiment polarity** — proportion of verdicts per category

These metrics allow researchers to assess whether the fact-checking corpus
reflects a deliberatively representative sample of political discourse.

Example usage
-------------
    from src.data.load_dataset import load_infact
    from src.data.label_mapping import apply_label_mapping
    from src.analysis.deliberation_metrics import run_deliberation_analysis

    df = load_infact("data/infact_dataset.tsv")
    df = apply_label_mapping(df)
    report = run_deliberation_analysis(df, output_dir="results/reports")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

logger = logging.getLogger(__name__)

FIGURE_DPI = 150


# ---------------------------------------------------------------------------
# Diversity / entropy helpers
# ---------------------------------------------------------------------------

def shannon_entropy(series: pd.Series) -> float:
    """Compute the Shannon entropy (base-2) of the value distribution.

    Parameters
    ----------
    series:
        Categorical series.

    Returns
    -------
    float
        Entropy in bits.  Returns 0.0 for empty or single-value series.
    """
    counts = series.value_counts()
    probs = counts / counts.sum()
    return float(scipy_entropy(probs, base=2))


def normalised_entropy(series: pd.Series) -> float:
    """Compute the normalised Shannon entropy (0 = uniform, 1 = maximally unequal).

    Note: normalised as H / log2(N) so higher values mean *more* uniform
    (higher deliberative diversity).

    Parameters
    ----------
    series:
        Categorical series.

    Returns
    -------
    float
        Value in [0, 1].
    """
    n_cats = series.nunique()
    if n_cats <= 1:
        return 0.0
    h = shannon_entropy(series)
    return float(h / np.log2(n_cats))


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def compute_voice_diversity(df: pd.DataFrame) -> dict:
    """Compute voice diversity metrics (authors and outlets).

    Parameters
    ----------
    df:
        INFACT DataFrame.

    Returns
    -------
    dict
        Counts and entropy measures for ``author_claim`` and ``source_outlet``.
    """
    result: dict = {}
    for col in ("author_claim", "source_outlet"):
        if col not in df.columns:
            continue
        series = df[col].dropna()
        result[col] = {
            "n_unique": int(series.nunique()),
            "top_10": series.value_counts().head(10).to_dict(),
            "shannon_entropy": shannon_entropy(series),
            "normalised_entropy": normalised_entropy(series),
        }
    return result


def compute_domain_coverage(df: pd.DataFrame) -> dict:
    """Compute domain coverage metrics.

    Parameters
    ----------
    df:
        INFACT DataFrame.

    Returns
    -------
    dict
        Domain counts, proportions, and entropy.
    """
    if "domain_claim" not in df.columns:
        return {}
    series = df["domain_claim"].dropna()
    counts = series.value_counts()
    return {
        "n_domains": int(series.nunique()),
        "counts": counts.to_dict(),
        "proportions": (counts / counts.sum()).round(4).to_dict(),
        "shannon_entropy": shannon_entropy(series),
        "normalised_entropy": normalised_entropy(series),
    }


def compute_temporal_spread(df: pd.DataFrame) -> dict:
    """Compute temporal spread metrics.

    Parameters
    ----------
    df:
        INFACT DataFrame with a ``date_verified`` datetime column.

    Returns
    -------
    dict
        Monthly counts, year distribution, and coverage span in days.
    """
    if "date_verified" not in df.columns:
        return {}
    valid = df["date_verified"].dropna()
    if valid.empty:
        return {}
    span_days = int((valid.max() - valid.min()).days)
    monthly = df.set_index("date_verified").resample("ME").size()
    yearly = df["date_verified"].dt.year.value_counts().sort_index()
    return {
        "span_days": span_days,
        "n_months_covered": int((monthly > 0).sum()),
        "yearly_counts": {int(k): int(v) for k, v in yearly.items()},
        "monthly_mean": float(monthly[monthly > 0].mean()),
        "monthly_std": float(monthly[monthly > 0].std()),
    }


def compute_verification_scope_distribution(df: pd.DataFrame) -> dict:
    """Compute distribution of verification scope values.

    Parameters
    ----------
    df:
        INFACT DataFrame.

    Returns
    -------
    dict
        Counts and proportions for ``verification_scope``.
    """
    if "verification_scope" not in df.columns:
        return {}
    series = df["verification_scope"].dropna()
    counts = series.value_counts()
    return {
        "counts": counts.to_dict(),
        "proportions": (counts / counts.sum()).round(4).to_dict(),
        "n_unique": int(series.nunique()),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_domain_entropy_heatmap(
    df: pd.DataFrame,
    output_dir: str = "results/figures",
    show: bool = False,
) -> Optional[Path]:
    """Plot a heatmap of verdict distribution per domain.

    Parameters
    ----------
    df:
        INFACT DataFrame (requires ``domain_claim`` and ``verdict_normalized``
        or ``verdict_original``).
    output_dir:
        Directory where the figure is saved.
    show:
        Whether to call ``plt.show()``.

    Returns
    -------
    Path | None
    """
    label_col = "verdict_normalized" if "verdict_normalized" in df.columns else "verdict_original"
    if "domain_claim" not in df.columns or label_col not in df.columns:
        logger.warning("Required columns missing for heatmap — skipping.")
        return None

    crosstab = pd.crosstab(df["domain_claim"], df[label_col], normalize="index").round(3)
    if crosstab.empty:
        return None

    # Limit to top 20 domains
    top_domains = df["domain_claim"].value_counts().head(20).index
    crosstab = crosstab.loc[crosstab.index.intersection(top_domains)]

    fig, ax = plt.subplots(figsize=(14, max(6, len(crosstab) * 0.5)))
    im = ax.imshow(crosstab.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(crosstab.columns)))
    ax.set_xticklabels(crosstab.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(crosstab.index)))
    ax.set_yticklabels(crosstab.index, fontsize=9)
    ax.set_title("Verdict Distribution per Domain (normalised by row)", fontsize=13)
    plt.colorbar(im, ax=ax, label="Proportion")
    plt.tight_layout()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_path = out / "domain_verdict_heatmap.png"
    fig.savefig(save_path, dpi=FIGURE_DPI)
    logger.info("Saved heatmap: %s", save_path)
    if show:
        plt.show()
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_deliberation_analysis(
    df: pd.DataFrame,
    output_dir: str = "results/reports",
    figure_dir: str = "results/figures",
    show: bool = False,
) -> dict:
    """Run the full deliberation metrics analysis and save results.

    Parameters
    ----------
    df:
        INFACT DataFrame (label mapping recommended).
    output_dir:
        Directory for JSON report.
    figure_dir:
        Directory for figures.
    show:
        Whether to display plots interactively.

    Returns
    -------
    dict
        Aggregated deliberation metrics report.
    """
    report = {
        "voice_diversity": compute_voice_diversity(df),
        "domain_coverage": compute_domain_coverage(df),
        "temporal_spread": compute_temporal_spread(df),
        "verification_scope": compute_verification_scope_distribution(df),
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "deliberation_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info("Saved deliberation report to %s", report_path)

    plot_domain_entropy_heatmap(df, output_dir=figure_dir, show=show)

    # Print summary
    print("\n--- Deliberation Analysis Summary ---")
    dc = report.get("domain_coverage", {})
    if dc:
        print(f"  Domains covered     : {dc.get('n_domains', 'N/A')}")
        print(f"  Domain entropy (H)  : {dc.get('shannon_entropy', 0):.3f} bits")
    ts = report.get("temporal_spread", {})
    if ts:
        print(f"  Corpus span         : {ts.get('span_days', 'N/A')} days")
    vd = report.get("voice_diversity", {})
    if "author_claim" in vd:
        print(f"  Unique authors      : {vd['author_claim']['n_unique']}")
    print()

    return report


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    from src.data.label_mapping import apply_label_mapping
    from src.data.load_dataset import load_infact, validate_dataset

    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/infact_dataset.tsv"
    dataset = load_infact(data_path)
    validate_dataset(dataset)
    dataset = apply_label_mapping(dataset)
    run_deliberation_analysis(dataset)
