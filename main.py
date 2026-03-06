#!/usr/bin/env python3
"""Main orchestration script for the RoFACT-Delib pipeline."""
import argparse
import logging

from src.utils.io import setup_logging, ensure_output_dirs, set_seeds
from src.data.load_and_validate import run_validation
from src.data.label_mapping import run_label_engineering
from src.eda.eda_report import run_eda
from src.tasks.claim_verification import run_baselines
from src.tasks.deliberation_analysis import run_deliberation_analysis
from src.tasks.linguistic_bias import run_linguistic_analysis
from src.tasks.ethics_audit import run_ethics_audit
from src.tasks.llm_experiments import run_llm_experiments, stratified_sample


def parse_args():
    parser = argparse.ArgumentParser(description="RoFACT-Delib Pipeline")
    parser.add_argument("--dataset", default="data/rofact.tsv")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-llm", action="store_true")
    parser.add_argument("--llm-backend", choices=["openai", "hf"], default="openai")
    parser.add_argument("--llm-sample-size", type=int, default=30)
    parser.add_argument("--hf-model", default="gpt2")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging("rofact_pipeline")
    set_seeds(args.seed)
    ensure_output_dirs(args.output_dir)

    logger.info("Task 1: Load and validate dataset")
    df, report = run_validation(args.dataset, args.output_dir)

    logger.info("Task 2: Label engineering")
    df = run_label_engineering(df, args.output_dir)

    logger.info("Task 3: EDA")
    run_eda(df, args.output_dir)

    logger.info("Task 4: Classical baselines")
    run_baselines(df, args.output_dir, args.seed)

    logger.info("Task 5: Deliberation analysis")
    run_deliberation_analysis(df, args.output_dir)

    logger.info("Task 6: Linguistic framing")
    run_linguistic_analysis(df, args.output_dir)

    logger.info("Task 7: Ethics audit")
    run_ethics_audit(df, args.output_dir)

    if args.run_llm:
        logger.info("Task 8: LLM experiments")
        import os
        if args.llm_backend == "openai":
            from src.tasks.llm_experiments import OpenAIBackend
            backend = OpenAIBackend(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                api_key=os.environ.get("OPENAI_API_KEY", ""),
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"),
            )
        else:
            from src.tasks.llm_experiments import HFBackend
            backend = HFBackend(model_name=args.hf_model)
        sample_df = stratified_sample(df, n=args.llm_sample_size, seed=args.seed)
        run_llm_experiments(sample_df, backend, args.output_dir, args.llm_sample_size, args.seed)

    logger.info("Pipeline complete. Outputs saved to %s/", args.output_dir)


if __name__ == "__main__":
    main()
