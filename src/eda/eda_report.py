"""Exploratory Data Analysis report generation for the RoFACT dataset."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import save_figure, save_markdown, save_table, setup_logging

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Tabular summaries
# ---------------------------------------------------------------------------


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of each verdict_original value."""
    counts = df["verdict_original"].value_counts().rename_axis("verdict_original").reset_index(name="count")
    counts["percent"] = (counts["count"] / counts["count"].sum() * 100).round(2)
    return counts


def epistemic_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of each epistemic_outcome value."""
    counts = df["epistemic_outcome"].value_counts().rename_axis("epistemic_outcome").reset_index(name="count")
    counts["percent"] = (counts["count"] / counts["count"].sum() * 100).round(2)
    return counts


def domain_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of each domain_claim value."""
    counts = df["domain_claim"].value_counts().rename_axis("domain_claim").reset_index(name="count")
    counts["percent"] = (counts["count"] / counts["count"].sum() * 100).round(2)
    return counts


def top_authors(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top-n authors by claim count."""
    counts = df["author_claim"].value_counts().head(n).rename_axis("author_claim").reset_index(name="count")
    return counts


def top_outlets(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top-n source outlets by claim count."""
    counts = df["source_outlet"].value_counts().head(n).rename_axis("source_outlet").reset_index(name="count")
    return counts


def section_length_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for word_count_* columns."""
    wc_cols = [c for c in df.columns if c.startswith("word_count_")]
    if not wc_cols:
        return pd.DataFrame()
    return df[wc_cols].describe().round(1)


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of missing (NaN or empty string) values per column."""
    rows = []
    for col in df.columns:
        n_nan = int(df[col].isna().sum())
        n_empty = int((df[col] == "").sum()) if df[col].dtype == object else 0
        total_missing = n_nan + n_empty
        rows.append(
            {
                "column": col,
                "missing_count": total_missing,
                "missing_percent": round(total_missing / len(df) * 100, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("missing_count", ascending=False).reset_index(drop=True)


def imbalance_ratio(df: pd.DataFrame, col: str) -> float:
    """Return max_count / min_count for a categorical column (>0 classes only)."""
    counts = df[col].value_counts()
    counts = counts[counts > 0]
    if len(counts) < 2:
        return 1.0
    return float(counts.iloc[0] / counts.iloc[-1])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_verdict_distribution(df: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart of verdict_original distribution."""
    data = label_distribution(df).sort_values("count")
    fig, ax = plt.subplots(figsize=(9, max(4, len(data) * 0.45)))
    bars = ax.barh(data["verdict_original"], data["count"], color="steelblue")
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_xlabel("Number of Claims")
    ax.set_title("Verdict Distribution (verdict_original)")
    plt.tight_layout()
    return fig


def plot_verification_length_hist(df: pd.DataFrame) -> plt.Figure:
    """Histogram of word_count_verification."""
    col = "word_count_verification"
    fig, ax = plt.subplots(figsize=(8, 4))
    if col in df.columns:
        data = df[col].dropna()
        ax.hist(data, bins=40, color="teal", edgecolor="white")
        ax.set_xlabel("Word Count (verification section)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Verification Section Length")
        mean_val = data.mean()
        ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean = {mean_val:.0f}")
        ax.legend()
    plt.tight_layout()
    return fig


def plot_domain_distribution(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of domain_claim distribution."""
    data = domain_distribution(df).sort_values("count", ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(4, len(data) * 0.4)))
    bars = ax.bar(data["domain_claim"], data["count"], color="coral")
    ax.bar_label(bars, fmt="%d", padding=2)
    ax.set_xlabel("Domain")
    ax.set_ylabel("Number of Claims")
    ax.set_title("Claim Distribution by Domain")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_top_authors(df: pd.DataFrame, n: int = 10) -> plt.Figure:
    """Horizontal bar chart of top-n authors by claim count."""
    data = top_authors(df, n=n).sort_values("count")
    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.5)))
    bars = ax.barh(data["author_claim"], data["count"], color="mediumpurple")
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_xlabel("Number of Claims")
    ax.set_title(f"Top {n} Authors by Claim Count")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Full EDA run
# ---------------------------------------------------------------------------


def run_eda(df: pd.DataFrame, output_dir: str = "results") -> str:
    """Generate all EDA tables, figures, and a Markdown summary report.

    Saves tables to results/tables/, figures to results/figures/, and the
    narrative report to results/reports/eda_report.md.

    Returns the Markdown report text.
    """
    tables_dir = Path(output_dir) / "tables"
    figures_dir = Path(output_dir) / "figures"
    reports_dir = Path(output_dir) / "reports"
    for d in (tables_dir, figures_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- Tables ---
    lbl = label_distribution(df)
    save_table(lbl, tables_dir / "verdict_distribution.csv")

    if "epistemic_outcome" in df.columns:
        epi = epistemic_distribution(df)
        save_table(epi, tables_dir / "epistemic_distribution.csv")
    else:
        epi = pd.DataFrame()

    dom = domain_distribution(df)
    save_table(dom, tables_dir / "domain_distribution.csv")

    auths = top_authors(df)
    save_table(auths, tables_dir / "top_authors.csv")

    outls = top_outlets(df)
    save_table(outls, tables_dir / "top_outlets.csv")

    miss = missingness_report(df)
    save_table(miss, tables_dir / "missingness_report.csv")

    sec_stats = section_length_stats(df)
    if not sec_stats.empty:
        save_table(sec_stats.reset_index(), tables_dir / "section_length_stats.csv")

    # --- Figures ---
    fig_verdict = plot_verdict_distribution(df)
    save_figure(fig_verdict, figures_dir / "verdict_distribution.png")
    plt.close(fig_verdict)

    fig_hist = plot_verification_length_hist(df)
    save_figure(fig_hist, figures_dir / "verification_length_hist.png")
    plt.close(fig_hist)

    fig_domain = plot_domain_distribution(df)
    save_figure(fig_domain, figures_dir / "domain_distribution.png")
    plt.close(fig_domain)

    fig_authors = plot_top_authors(df)
    save_figure(fig_authors, figures_dir / "top_authors.png")
    plt.close(fig_authors)

    # --- Markdown report ---
    imb_verdict = imbalance_ratio(df, "verdict_original")
    lines = [
        "# Exploratory Data Analysis Report\n\n",
        f"**Total records:** {len(df)}  \n",
        f"**Columns:** {len(df.columns)}\n\n",
        "## Verdict Distribution\n",
        lbl.to_markdown(index=False) + "\n\n",
        f"**Class imbalance ratio (verdict_original):** {imb_verdict:.1f}\n\n",
    ]

    if not epi.empty:
        imb_epi = imbalance_ratio(df, "epistemic_outcome")
        lines += [
            "## Epistemic Outcome Distribution\n",
            epi.to_markdown(index=False) + "\n\n",
            f"**Class imbalance ratio (epistemic_outcome):** {imb_epi:.1f}\n\n",
        ]

    lines += [
        "## Domain Distribution\n",
        dom.to_markdown(index=False) + "\n\n",
        "## Top 10 Authors\n",
        auths.to_markdown(index=False) + "\n\n",
        "## Top 10 Source Outlets\n",
        outls.to_markdown(index=False) + "\n\n",
        "## Missingness\n",
        miss[miss["missing_count"] > 0].to_markdown(index=False) + "\n\n"
        if not miss[miss["missing_count"] > 0].empty
        else "_No missing values detected._\n\n",
    ]

    if not sec_stats.empty:
        lines += [
            "## Section Length Statistics\n",
            sec_stats.to_markdown() + "\n\n",
        ]

    lines += [
        "## Figures Generated\n",
        "- `figures/verdict_distribution.png`\n",
        "- `figures/verification_length_hist.png`\n",
        "- `figures/domain_distribution.png`\n",
        "- `figures/top_authors.png`\n",
    ]

    report_text = "".join(lines)
    save_markdown(report_text, reports_dir / "eda_report.md")
    logger.info("EDA report saved to %s/reports/eda_report.md", output_dir)
    return report_text
