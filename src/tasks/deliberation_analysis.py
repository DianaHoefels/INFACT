"""Deliberation and concentration analysis for the RoFACT dataset."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io import save_figure, save_markdown, save_table, setup_logging

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Concentration metrics
# ---------------------------------------------------------------------------


def gini_coefficient(values) -> float:
    """Compute the Gini coefficient for an array of non-negative counts."""
    arr = np.array(values, dtype=float)
    arr = arr[arr >= 0]
    if len(arr) == 0 or arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return float((2 * (index * arr).sum() / (n * arr.sum())) - (n + 1) / n)


def top_share(values, n: int = 10) -> float:
    """Return the fraction of the total represented by the top-n values."""
    arr = np.array(values, dtype=float)
    total = arr.sum()
    if total == 0:
        return 0.0
    top = np.sort(arr)[::-1][:n].sum()
    return float(top / total)


# ---------------------------------------------------------------------------
# Author / domain / outlet analysis
# ---------------------------------------------------------------------------


def author_representation(df: pd.DataFrame) -> pd.DataFrame:
    """Claims per author with Gini and top-10 share summary."""
    counts = df["author_claim"].value_counts().reset_index()
    counts.columns = ["author_claim", "claim_count"]
    gini = gini_coefficient(counts["claim_count"].values)
    t10 = top_share(counts["claim_count"].values, n=10)
    counts["gini_all"] = round(gini, 4)
    counts["top10_share"] = round(t10, 4)
    return counts


def verdict_by_domain(df: pd.DataFrame) -> pd.DataFrame:
    """Normalised crosstab: domain vs epistemic_outcome (row-normalised)."""
    if "epistemic_outcome" not in df.columns:
        return pd.DataFrame()
    ct = pd.crosstab(df["domain_claim"], df["epistemic_outcome"], normalize="index").round(4)
    ct.index.name = "domain_claim"
    return ct.reset_index()


def verdict_by_author(df: pd.DataFrame, min_claims: int = 5) -> pd.DataFrame:
    """Row-normalised verdict distribution for authors with ≥ min_claims claims."""
    if "epistemic_outcome" not in df.columns:
        return pd.DataFrame()
    author_counts = df["author_claim"].value_counts()
    active_authors = author_counts[author_counts >= min_claims].index
    subset = df[df["author_claim"].isin(active_authors)]
    if subset.empty:
        return pd.DataFrame()
    ct = pd.crosstab(subset["author_claim"], subset["epistemic_outcome"], normalize="index").round(4)
    ct.index.name = "author_claim"
    return ct.reset_index()


def yearly_verdict_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Counts of epistemic_outcome per year_verified (NaT rows excluded)."""
    if "year_verified" not in df.columns or "epistemic_outcome" not in df.columns:
        return pd.DataFrame()
    valid = df[df["year_verified"].notna()].copy()
    valid["year_verified"] = valid["year_verified"].astype(int)
    ct = (
        valid.groupby(["year_verified", "epistemic_outcome"])
        .size()
        .reset_index(name="count")
    )
    return ct


def concentration_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary of Gini + top-10 share for authors, domains, and outlets."""
    rows = []
    for col in ("author_claim", "domain_claim", "source_outlet"):
        if col not in df.columns:
            continue
        counts = df[col].value_counts().values
        rows.append(
            {
                "dimension": col,
                "unique_values": int(df[col].nunique()),
                "total_claims": int(len(df)),
                "gini": round(gini_coefficient(counts), 4),
                "top10_share": round(top_share(counts, n=10), 4),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_yearly_trends(df: pd.DataFrame) -> plt.Figure:
    """Stacked bar chart of yearly verdict trends."""
    trends = yearly_verdict_trends(df)
    if trends.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No temporal data available", ha="center", va="center")
        return fig

    pivot = trends.pivot(index="year_verified", columns="epistemic_outcome", values="count").fillna(0)
    colors = {
        "true": "#2ecc71",
        "false": "#e74c3c",
        "partial": "#f39c12",
        "unverifiable": "#95a5a6",
        "other": "#bdc3c7",
        "unknown": "#ecf0f1",
    }
    col_colors = [colors.get(c, "#aaaaaa") for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot) * 0.8), 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=col_colors, edgecolor="white")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Claims")
    ax.set_title("Yearly Verdict Trends")
    ax.legend(title="Epistemic Outcome", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Full deliberation analysis run
# ---------------------------------------------------------------------------


def run_deliberation_analysis(df: pd.DataFrame, output_dir: str = "results") -> str:
    """Run all deliberation analyses, save outputs, return Markdown report.

    Saves:
      - results/tables/deliberation_summary.csv
      - results/tables/yearly_verdict_trends.csv
      - results/tables/verdict_by_domain.csv
      - results/figures/yearly_verdict_trends.png
      - results/reports/deliberation_report.md
    """
    tables_dir = Path(output_dir) / "tables"
    figures_dir = Path(output_dir) / "figures"
    reports_dir = Path(output_dir) / "reports"
    for d in (tables_dir, figures_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- Analyses ---
    conc = concentration_summary(df)
    save_table(conc, tables_dir / "deliberation_summary.csv")

    trends = yearly_verdict_trends(df)
    save_table(trends, tables_dir / "yearly_verdict_trends.csv")

    vbd = verdict_by_domain(df)
    save_table(vbd, tables_dir / "verdict_by_domain.csv")

    auth_rep = author_representation(df)
    save_table(auth_rep.head(50), tables_dir / "author_representation.csv")

    vba = verdict_by_author(df, min_claims=5)
    save_table(vba, tables_dir / "verdict_by_author.csv")

    # --- Figures ---
    fig_trends = plot_yearly_trends(df)
    save_figure(fig_trends, figures_dir / "yearly_verdict_trends.png")
    plt.close(fig_trends)

    # --- Report ---
    lines = [
        "# Deliberation Analysis Report\n\n",
        "## Concentration Summary\n",
        conc.to_markdown(index=False) + "\n\n",
        "### Interpretation\n",
        "A Gini coefficient close to 1 indicates high concentration "
        "(a small number of actors account for most of the claims). "
        "Top-10 share shows the fraction of claims held by the 10 most active entities.\n\n",
    ]

    if not trends.empty:
        years_covered = sorted(trends["year_verified"].unique())
        lines += [
            "## Yearly Verdict Trends\n",
            f"Data spans {min(years_covered)}–{max(years_covered)}.\n\n",
            trends.pivot(
                index="year_verified", columns="epistemic_outcome", values="count"
            ).fillna(0).astype(int).to_markdown() + "\n\n",
        ]

    if not vbd.empty:
        lines += [
            "## Verdict Distribution by Domain\n",
            "_Row-normalised (proportion within each domain)_\n\n",
            vbd.to_markdown(index=False) + "\n\n",
        ]

    if not vba.empty:
        lines += [
            f"## Verdict Distribution by Author (≥5 claims)\n",
            vba.head(20).to_markdown(index=False) + "\n\n",
        ]

    lines += ["## Figures\n", "- `figures/yearly_verdict_trends.png`\n"]

    report_text = "".join(lines)
    save_markdown(report_text, reports_dir / "deliberation_report.md")
    logger.info("Deliberation analysis saved to %s/", output_dir)
    return report_text
