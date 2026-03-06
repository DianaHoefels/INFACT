"""LLM experiment scaffolding for the RoFACT pipeline.

Supports OpenAI-compatible HTTP APIs (via urllib, no extra deps) and local
Hugging Face models (optional import).
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io import save_figure, save_markdown, save_table, setup_logging

logger = setup_logging(__name__)

TASK_VARIANTS: dict[str, str] = {
    "A": "zero_shot_verdict",
    "B": "few_shot_verdict",
    "C": "chain_of_thought",
    "D": "institutional_fidelity",
}

_VALID_LABELS_DEFAULT = ["true", "false", "partial", "unverifiable", "other"]


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class OpenAIBackend:
    """Thin HTTP wrapper for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1/chat/completions",
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        """Send *prompt* as a user message and return the assistant content string."""
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            self.base_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            logger.error("OpenAI API error: %s", exc)
            return ""


class HFBackend:
    """Local Hugging Face text-generation backend (optional dependency)."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        try:
            from transformers import pipeline as hf_pipeline  # noqa: PLC0415
            self._pipeline = hf_pipeline(
                "text-generation",
                model=model_name,
                device_map="auto",
                trust_remote_code=False,
            )
        except ImportError as exc:
            raise ImportError(
                "HFBackend requires `transformers` and `torch`. "
                "Install them with: pip install transformers torch"
            ) from exc

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate text continuation for *prompt* using the loaded HF model."""
        try:
            outputs = self._pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._pipeline.tokenizer.eos_token_id,
            )
            full_text: str = outputs[0]["generated_text"]
            # Return only the newly generated portion
            return full_text[len(prompt):].strip()
        except Exception as exc:  # noqa: BLE001
            logger.error("HFBackend generation error: %s", exc)
            return ""


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def load_prompts(path: str = "prompts/llm_prompts.md") -> dict[str, str]:
    """Parse a Markdown file into a dict mapping task-name → prompt template.

    Expected format::

        ## task_name
        <prompt text – may span multiple lines>

        ## another_task_name
        ...
    """
    prompts: dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Prompts file not found: %s. Using empty dict.", path)
        return prompts

    current_key: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines(keepends=True):
        m = re.match(r"^##\s+(\S+)", line)
        if m:
            if current_key is not None:
                prompts[current_key] = "".join(current_lines).strip()
            current_key = m.group(1)
            current_lines = []
        else:
            current_lines.append(line)

    if current_key is not None:
        prompts[current_key] = "".join(current_lines).strip()

    return prompts


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def stratified_sample(
    df: pd.DataFrame,
    n: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Draw a diversity-aware sample across domain and author dimensions.

    Falls back to random sampling when the dataset is too small for stratification.
    """
    if len(df) <= n:
        return df.copy()

    rng = np.random.default_rng(seed)

    groups = []
    strat_col = "domain_claim" if "domain_claim" in df.columns else None
    if strat_col is not None:
        domain_counts = df[strat_col].value_counts()
        n_domains = len(domain_counts)
        per_domain = max(1, n // n_domains)
        for domain, group in df.groupby(strat_col):
            sample_size = min(per_domain, len(group))
            groups.append(group.sample(n=sample_size, random_state=int(rng.integers(0, 10_000))))
        result = pd.concat(groups).drop_duplicates()
        if len(result) < n:
            remaining = df[~df.index.isin(result.index)]
            extra = remaining.sample(n=min(n - len(result), len(remaining)), random_state=seed)
            result = pd.concat([result, extra])
        return result.head(n).reset_index(drop=True)

    return df.sample(n=n, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Prompt formatting and response parsing
# ---------------------------------------------------------------------------


def format_prompt(template: str, row: pd.Series) -> str:
    """Fill *template* with values from *row* using {field} placeholders."""
    fields = {
        "claim_text": str(row.get("claim_text", "") or ""),
        "context": str(row.get("context", "") or ""),
        "verification": str(row.get("verification", "") or ""),
        "conclusion": str(row.get("conclusion", "") or ""),
        "domain_claim": str(row.get("domain_claim", "") or ""),
        "verdict_original": str(row.get("verdict_original", "") or ""),
        "author_claim": str(row.get("author_claim", "") or ""),
    }
    try:
        return template.format(**fields)
    except KeyError as exc:
        logger.warning("Missing placeholder in template: %s", exc)
        return template


def parse_verdict(response: str, valid_labels: list[str]) -> str:
    """Extract a verdict label from an LLM response string.

    Tries exact match, then case-insensitive substring search.
    Returns "unknown" if nothing parseable is found.
    """
    if not response:
        return "unknown"
    cleaned = response.strip().lower()
    for label in valid_labels:
        if label.lower() == cleaned:
            return label
    for label in valid_labels:
        if label.lower() in cleaned:
            return label
    return "unknown"


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def _plot_fidelity_bars(metrics_by_task: dict[str, dict], path: Path) -> None:
    """Save a grouped bar chart of macro-F1 by LLM task variant."""
    tasks = list(metrics_by_task.keys())
    f1_scores = [metrics_by_task[t].get("macro_f1", 0.0) for t in tasks]

    fig, ax = plt.subplots(figsize=(max(6, len(tasks) * 1.5), 4))
    bars = ax.bar(tasks, f1_scores, color="steelblue")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Task Variant")
    ax.set_ylabel("Macro F1")
    ax.set_title("LLM Experiment Fidelity – Macro F1 by Task")
    plt.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def run_llm_experiments(
    df: pd.DataFrame,
    backend,
    output_dir: str = "results",
    sample_size: int = 30,
    seed: int = 42,
) -> dict:
    """Run all LLM task variants (A–D), evaluate, and save outputs.

    Saves:
      - results/tables/llm_predictions.jsonl
      - results/tables/llm_metrics.csv
      - results/tables/llm_results_table.csv
      - results/figures/llm_fidelity_bars.png
      - results/reports/llm_report.md
      - results/reports/llm_qualitative_errors.md

    Returns a dict of {task_id: metrics_dict}.
    """
    tables_dir = Path(output_dir) / "tables"
    figures_dir = Path(output_dir) / "figures"
    reports_dir = Path(output_dir) / "reports"
    for d in (tables_dir, figures_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts("prompts/llm_prompts.md")
    valid_labels = _VALID_LABELS_DEFAULT

    all_predictions: list[dict] = []
    metrics_by_task: dict[str, dict] = {}
    errors_by_task: dict[str, list[dict]] = {}

    label_col = "epistemic_outcome" if "epistemic_outcome" in df.columns else "verdict_original"

    for task_id, task_name in TASK_VARIANTS.items():
        template = prompts.get(task_name, "")
        if not template:
            logger.warning("No prompt template found for task %s (%s). Skipping.", task_id, task_name)
            continue

        logger.info("Running LLM task %s: %s …", task_id, task_name)
        task_preds: list[str] = []
        task_truths: list[str] = []
        task_errors: list[dict] = []

        for _, row in df.iterrows():
            prompt = format_prompt(template, row)
            response = backend.generate(prompt, max_tokens=256, temperature=0.0) if hasattr(backend, "generate") else ""
            predicted = parse_verdict(response, valid_labels)
            truth = str(row.get(label_col, "unknown") or "unknown")

            record = {
                "task_id": task_id,
                "task_name": task_name,
                "record_id": str(row.get("record_id", "")),
                "prompt": prompt[:500],
                "response": response[:500],
                "predicted": predicted,
                "truth": truth,
            }
            all_predictions.append(record)
            task_preds.append(predicted)
            task_truths.append(truth)

            if predicted != truth and predicted != "unknown":
                task_errors.append(
                    {
                        "record_id": record["record_id"],
                        "truth": truth,
                        "predicted": predicted,
                        "response_snippet": response[:200],
                    }
                )

        errors_by_task[task_id] = task_errors

        # Compute metrics
        from src.utils.metrics import accuracy, macro_f1, weighted_f1  # noqa: PLC0415
        filtered_pairs = [
            (t, p) for t, p in zip(task_truths, task_preds) if t != "unknown"
        ]
        if filtered_pairs:
            y_true, y_pred = zip(*filtered_pairs)
            metrics_by_task[task_id] = {
                "task_name": task_name,
                "macro_f1": round(macro_f1(list(y_true), list(y_pred)), 4),
                "weighted_f1": round(weighted_f1(list(y_true), list(y_pred)), 4),
                "accuracy": round(accuracy(list(y_true), list(y_pred)), 4),
                "n_samples": len(y_true),
                "unknown_rate": round(task_preds.count("unknown") / max(len(task_preds), 1), 4),
            }
        else:
            metrics_by_task[task_id] = {
                "task_name": task_name,
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
                "accuracy": 0.0,
                "n_samples": 0,
                "unknown_rate": 1.0,
            }

    # Save JSONL predictions
    jsonl_path = tables_dir / "llm_predictions.jsonl"
    try:
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for rec in all_predictions:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.error("Failed to write llm_predictions.jsonl: %s", exc)

    # Save metrics CSV
    metrics_df = pd.DataFrame(metrics_by_task).T.reset_index().rename(columns={"index": "task_id"})
    save_table(metrics_df, tables_dir / "llm_metrics.csv")

    # Save results table (predictions)
    results_df = pd.DataFrame(all_predictions)[["task_id", "record_id", "truth", "predicted"]]
    save_table(results_df, tables_dir / "llm_results_table.csv")

    # Fidelity bar chart
    if metrics_by_task:
        _plot_fidelity_bars(metrics_by_task, figures_dir / "llm_fidelity_bars.png")

    # Markdown report
    report_lines = [
        "# LLM Experiment Report\n\n",
        f"**Sample size:** {len(df)}  \n",
        f"**Task variants:** {', '.join(TASK_VARIANTS.keys())}\n\n",
        "## Metrics by Task\n\n",
        metrics_df.to_markdown(index=False) + "\n\n" if not metrics_df.empty else "_No metrics._\n\n",
        "## Observations\n\n",
    ]
    for task_id, m in metrics_by_task.items():
        report_lines.append(
            f"- **Task {task_id} ({m['task_name']})**: Macro F1 = {m['macro_f1']:.4f}, "
            f"Accuracy = {m['accuracy']:.4f}, Unknown rate = {m['unknown_rate']:.2%}\n"
        )
    report_lines.append("\n## Figures\n- `figures/llm_fidelity_bars.png`\n")
    save_markdown("".join(report_lines), reports_dir / "llm_report.md")

    # Qualitative errors report
    error_lines = ["# LLM Qualitative Error Analysis\n\n"]
    for task_id, errs in errors_by_task.items():
        error_lines.append(f"## Task {task_id} – {TASK_VARIANTS.get(task_id, '')}\n\n")
        if not errs:
            error_lines.append("_No errors recorded._\n\n")
            continue
        for e in errs[:20]:
            error_lines.append(
                f"**Record:** {e['record_id']}  \n"
                f"**Truth:** {e['truth']} | **Predicted:** {e['predicted']}  \n"
                f"**Response snippet:** {e['response_snippet']}\n\n---\n\n"
            )
    save_markdown("".join(error_lines), reports_dir / "llm_qualitative_errors.md")

    logger.info("LLM experiments complete. %d task variants run.", len(metrics_by_task))
    return metrics_by_task
