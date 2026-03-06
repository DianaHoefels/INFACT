"""Linguistic framing and bias analysis for the RoFACT dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import save_markdown, save_table, setup_logging
from src.utils.text import (
    AUTHORITY_MARKERS,
    CERTAINTY_MARKERS,
    HEDGE_MARKERS,
    MODAL_MARKERS,
    compute_marker_rates,
)

logger = setup_logging(__name__)

_FEATURE_COLS = [
    "hedge_rate",
    "certainty_rate",
    "authority_rate",
    "modal_rate",
]


# ---------------------------------------------------------------------------
# Row-level feature extraction
# ---------------------------------------------------------------------------


def compute_text_features(row: pd.Series) -> dict:
    """Compute marker rates for all lexicon types from a DataFrame row.

    Uses claim_text + context + verification for a richer text signal.
    """
    text_parts = []
    for field in ("claim_text", "context", "verification"):
        val = row.get(field, "")
        if isinstance(val, str) and val.strip():
            text_parts.append(val.strip())
    combined = " ".join(text_parts)

    return {
        "hedge_rate": compute_marker_rates(combined, HEDGE_MARKERS),
        "certainty_rate": compute_marker_rates(combined, CERTAINTY_MARKERS),
        "authority_rate": compute_marker_rates(combined, AUTHORITY_MARKERS),
        "modal_rate": compute_marker_rates(combined, MODAL_MARKERS),
    }


def compute_linguistic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply compute_text_features to every row; return feature DataFrame."""
    features = df.apply(compute_text_features, axis=1, result_type="expand")
    features.index = df.index
    return features


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def aggregate_by_group(
    df: pd.DataFrame,
    feature_df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """Return mean feature rates grouped by *group_col*."""
    if group_col not in df.columns:
        return pd.DataFrame()
    combined = df[[group_col]].copy()
    for col in _FEATURE_COLS:
        combined[col] = feature_df[col].values
    agg = combined.groupby(group_col)[_FEATURE_COLS].mean().round(4).reset_index()
    count = df[group_col].value_counts().rename_axis(group_col).reset_index(name="claim_count")
    return agg.merge(count, on=group_col).sort_values("claim_count", ascending=False)


def rank_groups(agg_df: pd.DataFrame, feature_col: str) -> pd.DataFrame:
    """Sort a group-aggregated DataFrame by *feature_col* descending."""
    if feature_col not in agg_df.columns:
        return agg_df
    return agg_df.sort_values(feature_col, ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plain-language interpretation
# ---------------------------------------------------------------------------


def generate_interpretation(agg_df: pd.DataFrame) -> str:
    """Generate a plain-language paragraph summarising hedging and certainty patterns."""
    if agg_df.empty or "hedge_rate" not in agg_df.columns:
        return "Insufficient data for linguistic interpretation."

    group_col = agg_df.columns[0]

    top_hedge = rank_groups(agg_df, "hedge_rate").head(3)[group_col].tolist()
    top_certain = rank_groups(agg_df, "certainty_rate").head(3)[group_col].tolist()
    top_authority = rank_groups(agg_df, "authority_rate").head(3)[group_col].tolist()

    return (
        f"Among the groups analysed by `{group_col}`, the entities showing the highest hedging "
        f"language rates are: **{', '.join(str(x) for x in top_hedge)}**. "
        f"Conversely, the most certain language is used by: **{', '.join(str(x) for x in top_certain)}**. "
        f"High authority citation rates are observed in: **{', '.join(str(x) for x in top_authority)}**. "
        "These patterns may reflect editorial style, domain conventions, or systemic framing differences "
        "that warrant further investigation."
    )


# ---------------------------------------------------------------------------
# Full linguistic analysis run
# ---------------------------------------------------------------------------


def run_linguistic_analysis(df: pd.DataFrame, output_dir: str = "results") -> str:
    """Compute features, aggregate by groups, save tables and a Markdown report.

    Saves:
      - results/tables/linguistic_group_stats_by_author.csv
      - results/tables/linguistic_group_stats_by_outlet.csv
      - results/tables/linguistic_group_stats_by_domain.csv
      - results/reports/linguistic_report.md

    Returns the Markdown report text.
    """
    tables_dir = Path(output_dir) / "tables"
    reports_dir = Path(output_dir) / "reports"
    for d in (tables_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger.info("Computing linguistic features for %d rows…", len(df))
    feature_df = compute_linguistic_features(df)

    report_lines = ["# Linguistic Framing Analysis Report\n\n"]

    # Corpus-level summary
    corpus_means = feature_df[_FEATURE_COLS].mean().round(4)
    report_lines += [
        "## Corpus-Level Marker Rates (per 100 words)\n",
        corpus_means.to_markdown() + "\n\n",
    ]

    groups: dict[str, str] = {
        "author_claim": "by_author",
        "source_outlet": "by_outlet",
        "domain_claim": "by_domain",
    }

    for group_col, suffix in groups.items():
        agg = aggregate_by_group(df, feature_df, group_col)
        fname = tables_dir / f"linguistic_group_stats_{suffix}.csv"
        save_table(agg, fname)

        interpretation = generate_interpretation(agg)
        report_lines += [
            f"## Group Analysis – {group_col}\n\n",
            interpretation + "\n\n",
            "### Top 15 by Hedge Rate\n",
            rank_groups(agg, "hedge_rate").head(15).to_markdown(index=False) + "\n\n",
            "### Top 15 by Certainty Rate\n",
            rank_groups(agg, "certainty_rate").head(15).to_markdown(index=False) + "\n\n",
        ]

    # Aggregate by epistemic outcome if available
    if "epistemic_outcome" in df.columns:
        agg_outcome = aggregate_by_group(df, feature_df, "epistemic_outcome")
        report_lines += [
            "## Linguistic Features by Epistemic Outcome\n\n",
            agg_outcome.to_markdown(index=False) + "\n\n",
        ]

    report_text = "".join(report_lines)
    save_markdown(report_text, reports_dir / "linguistic_report.md")
    logger.info("Linguistic analysis report saved to %s/", output_dir)
    return report_text
