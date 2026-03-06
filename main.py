"""
main.py
-------
CLI entry point for the INFACT research pipeline.

Provides subcommands to run individual modules or the full pipeline:

    python main.py stats        — corpus statistics (EDA)
    python main.py baseline     — baseline claim verification
    python main.py llm          — LLM-based claim verification
    python main.py deliberation — deliberation-aware analysis
    python main.py linguistic   — linguistic framing analysis
    python main.py ethics       — bias / ethics audit
    python main.py all          — run all of the above

Common options
--------------
    --data_path     Path to the INFACT TSV dataset file
                    (default: data/infact_dataset.tsv)
    --output_dir    Root output directory (default: results)
    --log_level     Logging verbosity (default: INFO)

Example
-------
    python main.py all --data_path data/infact_dataset.tsv
    python main.py baseline --data_path data/infact_dataset.tsv
    python main.py llm --data_path data/infact_dataset.tsv \\
                       --model_name bert-base-multilingual-cased
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_data(data_path: str):
    """Load and label-map the INFACT dataset."""
    from src.data.label_mapping import apply_label_mapping
    from src.data.load_dataset import load_infact, validate_dataset

    df = load_infact(data_path)
    validate_dataset(df)
    df = apply_label_mapping(df)
    return df


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_stats(args: argparse.Namespace) -> None:
    """Run exploratory data analysis and save figures."""
    from src.eda.corpus_statistics import run_eda

    df = _load_data(args.data_path)
    run_eda(df, output_dir=str(Path(args.output_dir) / "figures"))


def cmd_baseline(args: argparse.Namespace) -> None:
    """Run TF-IDF + ML baseline verification experiments."""
    from src.experiments.baseline_verification import run_baseline

    df = _load_data(args.data_path)
    run_baseline(
        df,
        use_context=args.use_context,
        n_splits=args.n_splits,
        output_dir=str(Path(args.output_dir) / "tables"),
    )


def cmd_llm(args: argparse.Namespace) -> None:
    """Fine-tune a multilingual LM for claim verification."""
    from src.experiments.llm_verification import run_finetuning

    df = _load_data(args.data_path)
    run_finetuning(
        df,
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


def cmd_deliberation(args: argparse.Namespace) -> None:
    """Run deliberation-aware discourse analysis."""
    from src.analysis.deliberation_metrics import run_deliberation_analysis

    df = _load_data(args.data_path)
    run_deliberation_analysis(
        df,
        output_dir=str(Path(args.output_dir) / "reports"),
        figure_dir=str(Path(args.output_dir) / "figures"),
    )


def cmd_linguistic(args: argparse.Namespace) -> None:
    """Run linguistic framing analysis."""
    from src.analysis.linguistic_bias import run_linguistic_analysis

    df = _load_data(args.data_path)
    run_linguistic_analysis(
        df,
        output_dir=str(Path(args.output_dir) / "reports"),
        figure_dir=str(Path(args.output_dir) / "figures"),
    )


def cmd_ethics(args: argparse.Namespace) -> None:
    """Run bias / ethics audit."""
    from src.analysis.ethics_audit import run_ethics_audit

    df = _load_data(args.data_path)
    run_ethics_audit(
        df,
        output_dir=str(Path(args.output_dir) / "reports"),
    )


def cmd_all(args: argparse.Namespace) -> None:
    """Run all pipeline stages sequentially."""
    logging.getLogger(__name__).info("Running full INFACT pipeline …")
    cmd_stats(args)
    cmd_baseline(args)
    cmd_deliberation(args)
    cmd_linguistic(args)
    cmd_ethics(args)
    logging.getLogger(__name__).info("Full pipeline complete.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infact",
        description="INFACT: Romanian Institutional Fact-Checking NLP Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Global options
    parser.add_argument(
        "--data_path",
        default="data/infact_dataset.tsv",
        help="Path to the INFACT TSV dataset file.",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Root directory for saving results, figures, and reports.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- stats ---
    subparsers.add_parser(
        "stats",
        help="Run exploratory data analysis and generate corpus statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- baseline ---
    bp = subparsers.add_parser(
        "baseline",
        help="Run TF-IDF + ML baseline verification experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    bp.add_argument(
        "--use_context",
        action="store_true",
        help="Concatenate context column to claim text.",
    )
    bp.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )

    # --- llm ---
    lp = subparsers.add_parser(
        "llm",
        help="Fine-tune a multilingual LM for claim verification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    lp.add_argument(
        "--model_name",
        default="bert-base-multilingual-cased",
        help="HuggingFace model identifier.",
    )
    lp.add_argument("--num_epochs", type=int, default=3, help="Training epochs.")
    lp.add_argument("--batch_size", type=int, default=16, help="Batch size.")

    # --- deliberation ---
    subparsers.add_parser(
        "deliberation",
        help="Run deliberation-aware discourse analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- linguistic ---
    subparsers.add_parser(
        "linguistic",
        help="Run linguistic framing analysis (hedges, certainty, authority).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- ethics ---
    subparsers.add_parser(
        "ethics",
        help="Run bias and ethics audit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- all ---
    ap = subparsers.add_parser(
        "all",
        help="Run the full pipeline (stats → baseline → deliberation → linguistic → ethics).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--use_context",
        action="store_true",
        help="Pass --use_context to the baseline stage.",
    )
    ap.add_argument("--n_splits", type=int, default=5, help="CV folds for baseline.")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

COMMAND_MAP = {
    "stats": cmd_stats,
    "baseline": cmd_baseline,
    "llm": cmd_llm,
    "deliberation": cmd_deliberation,
    "linguistic": cmd_linguistic,
    "ethics": cmd_ethics,
    "all": cmd_all,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.log_level)

    handler = COMMAND_MAP.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    try:
        handler(args)
    except FileNotFoundError as exc:
        logging.error("File not found: %s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001
        logging.exception("Unexpected error: %s", exc)
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
