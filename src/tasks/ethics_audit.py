"""Ethics audit for the RoFACT dataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import save_markdown, save_table, setup_logging
from src.utils.text import CERTAINTY_MARKERS, HEDGE_MARKERS, compute_marker_rates

logger = setup_logging(__name__)

_SEVERITY_LEVELS = ("high", "medium", "low")

# Thresholds for flagging imbalance
_IMBALANCE_RATIO_HIGH = 20.0
_IMBALANCE_RATIO_MEDIUM = 10.0
_COVERAGE_DROP_THRESHOLD = 0.5  # >50% year-over-year drop flags temporal bias


# ---------------------------------------------------------------------------
# Audit functions
# ---------------------------------------------------------------------------


def representation_bias(df: pd.DataFrame) -> dict:
    """Check author, domain, outlet distribution for significant imbalances."""
    findings: dict[str, dict] = {}
    for col in ("author_claim", "domain_claim", "source_outlet"):
        if col not in df.columns:
            continue
        counts = df[col].value_counts()
        if len(counts) < 2:
            continue
        ratio = float(counts.iloc[0] / counts.iloc[-1])
        n_unique = int(counts.nunique())
        top5_share = float(counts.head(5).sum() / counts.sum())
        severity = (
            "high" if ratio >= _IMBALANCE_RATIO_HIGH
            else "medium" if ratio >= _IMBALANCE_RATIO_MEDIUM
            else "low"
        )
        findings[col] = {
            "n_unique": n_unique,
            "imbalance_ratio": round(ratio, 2),
            "top5_share": round(top5_share, 4),
            "most_common": str(counts.index[0]),
            "most_common_count": int(counts.iloc[0]),
            "severity": severity,
        }
    return findings


def outcome_bias(df: pd.DataFrame) -> dict:
    """Check verdict distribution by author and domain for disparities."""
    findings: dict[str, dict] = {}
    if "epistemic_outcome" not in df.columns:
        return findings
    for col in ("author_claim", "domain_claim"):
        if col not in df.columns:
            continue
        ct = pd.crosstab(df[col], df["epistemic_outcome"], normalize="index")
        if ct.empty or "false" not in ct.columns:
            continue
        # Find entities with highest 'false' rate (potential structural bias)
        false_col = ct["false"].sort_values(ascending=False)
        top_false = false_col.head(5)
        findings[col] = {
            "top5_highest_false_rate": top_false.round(4).to_dict(),
            "mean_false_rate": round(float(false_col.mean()), 4),
            "std_false_rate": round(float(false_col.std()), 4),
            "severity": "medium" if float(false_col.std()) > 0.2 else "low",
        }
    return findings


def linguistic_bias_check(df: pd.DataFrame) -> dict:
    """Check if hedge/certainty rates differ systematically by epistemic outcome."""
    findings: dict = {}
    if "epistemic_outcome" not in df.columns:
        return findings

    text_col = None
    for c in ("verification", "claim_text", "context"):
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        return findings

    df_work = df[["epistemic_outcome", text_col]].dropna().copy()
    df_work["hedge_rate"] = df_work[text_col].apply(lambda t: compute_marker_rates(t, HEDGE_MARKERS))
    df_work["certainty_rate"] = df_work[text_col].apply(lambda t: compute_marker_rates(t, CERTAINTY_MARKERS))

    agg = df_work.groupby("epistemic_outcome")[["hedge_rate", "certainty_rate"]].mean().round(4)
    hedge_range = float(agg["hedge_rate"].max() - agg["hedge_rate"].min())
    certainty_range = float(agg["certainty_rate"].max() - agg["certainty_rate"].min())

    findings = {
        "hedge_rate_by_outcome": agg["hedge_rate"].to_dict(),
        "certainty_rate_by_outcome": agg["certainty_rate"].to_dict(),
        "hedge_range": round(hedge_range, 4),
        "certainty_range": round(certainty_range, 4),
        "severity": "medium" if hedge_range > 1.0 or certainty_range > 1.0 else "low",
    }
    return findings


def temporal_bias(df: pd.DataFrame) -> dict:
    """Check claim coverage by year for gaps or dramatic drops."""
    findings: dict = {}
    if "year_verified" not in df.columns:
        return findings
    year_counts = (
        df[df["year_verified"].notna()]
        .groupby(df["year_verified"].astype(int))
        .size()
        .sort_index()
    )
    if len(year_counts) < 2:
        return {"note": "insufficient temporal data", "severity": "low"}

    year_list = year_counts.index.tolist()
    # Check for missing years
    expected_years = set(range(min(year_list), max(year_list) + 1))
    missing_years = sorted(expected_years - set(year_list))

    # Check for large year-over-year drops
    pct_changes = year_counts.pct_change().dropna()
    severe_drops = pct_changes[pct_changes < -_COVERAGE_DROP_THRESHOLD].to_dict()

    severity = "high" if missing_years or len(severe_drops) > 1 else "low"
    findings = {
        "years_covered": year_list,
        "missing_years": missing_years,
        "severe_drops": {str(k): round(float(v), 4) for k, v in severe_drops.items()},
        "min_year": int(min(year_list)),
        "max_year": int(max(year_list)),
        "severity": severity,
    }
    return findings


# ---------------------------------------------------------------------------
# Warning compilation and recommendations
# ---------------------------------------------------------------------------


def compile_warnings(findings: dict) -> list[dict]:
    """Flatten the findings dict into a list of warning dicts sorted by severity."""
    severity_order = {"high": 0, "medium": 1, "low": 2}
    warnings = []

    def _add(domain: str, sub: str, details: dict) -> None:
        severity = details.get("severity", "low")
        warnings.append(
            {
                "domain": domain,
                "sub_key": sub,
                "severity": severity,
                "details": {k: v for k, v in details.items() if k != "severity"},
            }
        )

    for domain, domain_findings in findings.items():
        if isinstance(domain_findings, dict):
            if "severity" in domain_findings:
                _add(domain, domain, domain_findings)
            else:
                for sub, sub_findings in domain_findings.items():
                    if isinstance(sub_findings, dict) and "severity" in sub_findings:
                        _add(domain, sub, sub_findings)

    return sorted(warnings, key=lambda w: severity_order.get(w["severity"], 3))


def mitigation_recommendations(warnings: list[dict]) -> list[str]:
    """Return a list of plain-language mitigation recommendation strings."""
    recommendations = []
    seen_domains = set()

    domain_recs: dict[str, str] = {
        "representation_bias": (
            "Consider over-sampling under-represented authors/domains or applying "
            "class-weighted models to address representation imbalance."
        ),
        "outcome_bias": (
            "Investigate structural factors driving high 'false' rates for specific actors. "
            "Stratify model evaluation by domain and author to detect systematic disparities."
        ),
        "linguistic_bias_check": (
            "Linguistic markers differ across verdict categories. Consider controlling for "
            "framing effects when building classifiers, or use debiasing techniques."
        ),
        "temporal_bias": (
            "Temporal coverage gaps may introduce recency bias. Consider temporal "
            "cross-validation and report model performance per time period."
        ),
    }

    for w in warnings:
        domain = w["domain"]
        if domain not in seen_domains:
            seen_domains.add(domain)
            if domain in domain_recs:
                prefix = "⚠️ [HIGH]" if w["severity"] == "high" else "ℹ️ [MEDIUM]" if w["severity"] == "medium" else "💡 [LOW]"
                recommendations.append(f"{prefix} {domain_recs[domain]}")

    if not recommendations:
        recommendations.append("No significant ethical concerns detected at current thresholds.")
    return recommendations


# ---------------------------------------------------------------------------
# Full audit run
# ---------------------------------------------------------------------------


def run_ethics_audit(df: pd.DataFrame, output_dir: str = "results") -> str:
    """Run all ethics audit functions, save results, return Markdown report.

    Saves:
      - results/tables/ethics_audit_summary.csv
      - results/reports/ethics_audit_report.md
    """
    tables_dir = Path(output_dir) / "tables"
    reports_dir = Path(output_dir) / "reports"
    for d in (tables_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    all_findings: dict[str, dict] = {}

    logger.info("Running representation bias audit…")
    rep_bias = representation_bias(df)
    all_findings["representation_bias"] = rep_bias

    logger.info("Running outcome bias audit…")
    out_bias = outcome_bias(df)
    all_findings["outcome_bias"] = out_bias

    logger.info("Running linguistic bias check…")
    ling_bias = linguistic_bias_check(df)
    all_findings["linguistic_bias_check"] = ling_bias

    logger.info("Running temporal bias audit…")
    temp_bias = temporal_bias(df)
    all_findings["temporal_bias"] = temp_bias

    warnings = compile_warnings(all_findings)
    recommendations = mitigation_recommendations(warnings)

    # Flatten to CSV
    summary_rows = []
    for w in warnings:
        summary_rows.append(
            {
                "domain": w["domain"],
                "sub_key": w["sub_key"],
                "severity": w["severity"],
                "details_summary": str(w["details"])[:300],
            }
        )
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        save_table(summary_df, tables_dir / "ethics_audit_summary.csv")

    # Markdown report
    lines = [
        "# Ethics Audit Report\n\n",
        f"**Total warnings:** {len(warnings)}  \n",
        f"**High severity:** {sum(1 for w in warnings if w['severity'] == 'high')}  \n",
        f"**Medium severity:** {sum(1 for w in warnings if w['severity'] == 'medium')}  \n\n",
    ]

    lines.append("## Representation Bias\n\n")
    if rep_bias:
        for col, info in rep_bias.items():
            lines.append(
                f"**{col}**: {info['n_unique']} unique values, "
                f"imbalance ratio = {info['imbalance_ratio']}, "
                f"top-5 share = {info['top5_share']:.1%}, "
                f"severity = **{info['severity']}**  \n"
                f"Most common: *{info['most_common']}* ({info['most_common_count']} claims)\n\n"
            )
    else:
        lines.append("No representation bias findings.\n\n")

    lines.append("## Outcome Bias\n\n")
    if out_bias:
        for col, info in out_bias.items():
            lines.append(
                f"**{col}**: mean false rate = {info['mean_false_rate']}, "
                f"std = {info['std_false_rate']}, severity = **{info['severity']}**\n\n"
                f"Top entities with highest 'false' classification rate:\n"
            )
            for entity, rate in info["top5_highest_false_rate"].items():
                lines.append(f"  - {entity}: {rate:.1%}\n")
            lines.append("\n")
    else:
        lines.append("No outcome bias findings.\n\n")

    lines.append("## Linguistic Bias\n\n")
    if ling_bias and "hedge_rate_by_outcome" in ling_bias:
        lines.append(
            f"Hedge rate range across outcomes: {ling_bias['hedge_range']:.4f}  \n"
            f"Certainty rate range: {ling_bias['certainty_range']:.4f}  \n"
            f"Severity: **{ling_bias['severity']}**\n\n"
        )
        lines.append("Hedge rate by outcome:\n")
        for outcome, rate in ling_bias["hedge_rate_by_outcome"].items():
            lines.append(f"  - {outcome}: {rate:.4f}\n")
        lines.append("\n")
    else:
        lines.append("No linguistic bias findings.\n\n")

    lines.append("## Temporal Bias\n\n")
    if temp_bias and "years_covered" in temp_bias:
        lines.append(
            f"Years covered: {temp_bias['min_year']}–{temp_bias['max_year']}  \n"
            f"Missing years: {temp_bias['missing_years'] or 'none'}  \n"
            f"Severity: **{temp_bias['severity']}**\n\n"
        )
    else:
        lines.append("Insufficient temporal data.\n\n")

    lines.append("## Warnings Summary\n\n")
    if warnings:
        lines.append("| Domain | Sub-key | Severity |\n|---|---|---|\n")
        for w in warnings:
            lines.append(f"| {w['domain']} | {w['sub_key']} | **{w['severity']}** |\n")
        lines.append("\n")

    lines.append("## Mitigation Recommendations\n\n")
    for rec in recommendations:
        lines.append(f"- {rec}\n")

    report_text = "".join(lines)
    save_markdown(report_text, reports_dir / "ethics_audit_report.md")
    logger.info("Ethics audit saved to %s/", output_dir)
    return report_text
