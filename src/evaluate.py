"""Evaluation script for INFACT fact-checking predictions.

Usage
-----
Evaluate a predictions file against the gold labels::

    python src/evaluate.py \\
        --gold data/test.jsonl \\
        --predictions predictions.jsonl

The predictions file must be a JSON Lines file where each line contains an
object with at least the fields ``id`` and ``label``.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

# Allow running as `python src/evaluate.py` from the repo root *and* as an
# installed module.  Insert the directory containing this file so that the
# sibling `data_utils` module is always importable.
sys.path.insert(0, str(Path(__file__).parent))
from data_utils import LABELS, load_jsonl  # noqa: E402


class EvaluationResult(NamedTuple):
    """Container for evaluation metrics."""

    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return *numerator / denominator*, or 0.0 when *denominator* is zero."""
    return numerator / denominator if denominator > 0 else 0.0


def evaluate(gold: list[dict], predictions: list[dict]) -> EvaluationResult:
    """Compute accuracy and macro-averaged F1 for INFACT predictions.

    Args:
        gold: List of gold-standard instances loaded from a split file.
            Each instance must have ``id`` and ``label`` fields.
        predictions: List of predicted instances. Each must have ``id``
            and ``label`` fields.

    Returns:
        An :class:`EvaluationResult` with accuracy, macro-F1, and
        per-class precision / recall / F1 scores.

    Raises:
        ValueError: If a prediction ID does not appear in the gold set.
    """
    gold_by_id = {inst["id"]: inst["label"] for inst in gold}

    true_positives: dict[str, int] = {label: 0 for label in LABELS}
    false_positives: dict[str, int] = {label: 0 for label in LABELS}
    false_negatives: dict[str, int] = {label: 0 for label in LABELS}

    correct = 0
    total = 0

    for pred_inst in predictions:
        pred_id = pred_inst["id"]
        if pred_id not in gold_by_id:
            raise ValueError(f"Prediction ID '{pred_id}' not found in gold data.")
        gold_label = gold_by_id[pred_id]
        pred_label = pred_inst["label"]

        if gold_label == pred_label:
            correct += 1
            true_positives[pred_label] = true_positives.get(pred_label, 0) + 1
        else:
            false_positives[pred_label] = false_positives.get(pred_label, 0) + 1
            false_negatives[gold_label] = false_negatives.get(gold_label, 0) + 1

        total += 1

    accuracy = _safe_divide(correct, total)

    per_class_precision: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}
    per_class_f1: dict[str, float] = {}

    for label in LABELS:
        tp = true_positives[label]
        fp = false_positives[label]
        fn = false_negatives[label]

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)

        per_class_precision[label] = precision
        per_class_recall[label] = recall
        per_class_f1[label] = f1

    macro_f1 = sum(per_class_f1.values()) / len(LABELS)

    return EvaluationResult(
        accuracy=accuracy,
        macro_f1=macro_f1,
        per_class_f1=per_class_f1,
        per_class_precision=per_class_precision,
        per_class_recall=per_class_recall,
    )


def print_results(result: EvaluationResult) -> None:
    """Print evaluation results to stdout in a human-readable table."""
    print("\n" + "=" * 55)
    print(f"{'INFACT Evaluation Results':^55}")
    print("=" * 55)
    print(f"{'Accuracy':<30} {result.accuracy * 100:>8.2f}%")
    print(f"{'Macro-F1':<30} {result.macro_f1 * 100:>8.2f}%")
    print("-" * 55)
    print(f"{'Label':<20} {'P':>8} {'R':>8} {'F1':>8}")
    print("-" * 55)
    for label in LABELS:
        p = result.per_class_precision[label] * 100
        r = result.per_class_recall[label] * 100
        f = result.per_class_f1[label] * 100
        print(f"{label:<20} {p:>8.2f} {r:>8.2f} {f:>8.2f}")
    print("=" * 55 + "\n")


def main(argv: list[str] | None = None) -> int:
    """Entry point for the evaluation script.

    Args:
        argv: Command-line arguments (defaults to ``sys.argv``).

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate INFACT fact-checking predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--gold",
        required=True,
        type=Path,
        help="Path to the gold-standard JSONL file (e.g. data/test.jsonl).",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Path to the predictions JSONL file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write evaluation results as JSON.",
    )

    args = parser.parse_args(argv)

    try:
        gold = load_jsonl(args.gold)
        predictions = load_jsonl(args.predictions)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        result = evaluate(gold, predictions)
    except ValueError as exc:
        print(f"Error during evaluation: {exc}", file=sys.stderr)
        return 1

    print_results(result)

    if args.output_json is not None:
        output = {
            "accuracy": result.accuracy,
            "macro_f1": result.macro_f1,
            "per_class_f1": result.per_class_f1,
            "per_class_precision": result.per_class_precision,
            "per_class_recall": result.per_class_recall,
        }
        args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Results written to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
