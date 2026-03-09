# INFACT AI coding guide

## Big picture
- `main.py` is the CLI entry point; it wires data loading → label mapping → experiment/analysis modules.
- The core data flow is `load_infact()` + `validate_dataset()` in `src/data/load_dataset.py`, then `apply_label_mapping()` in `src/data/label_mapping.py` to add `verdict_normalized`, `label_id`, and `label_binary` (rows mapped to `Other` are dropped by default).
- Outputs are written to `results/figures/`, `results/tables/`, and `results/reports/`; most modules create their own subfolders with `Path(...).mkdir(parents=True, exist_ok=True)`.

## Architecture map (by folder)
- `src/data/`: dataset loading, validation, and label engineering.
- `src/experiments/`: baseline ML (`baseline_verification.py`) and LLM fine-tuning/zero-shot (`llm_verification.py`).
- `src/analysis/`: deliberation metrics, linguistic framing, and ethics audits.
- `src/utils/`: shared I/O helpers and metric utilities.

## Project-specific conventions
- The TSV dataset must include the columns listed in `REQUIRED_COLUMNS` in `src/data/load_dataset.py`; `date_verified` is parsed with format `DD.MM.YYYY`.
- Label normalization is a Romanian-focused mapping; update `VERDICT_NORMALIZATION` when new verdict strings appear.
- Baseline experiments rely on `label_id` and `claim_text`; use `build_text_features()` if you add new text sources (it optionally concatenates `context`).
- Oversampling is only applied within CV training folds in `run_cross_validation()` to avoid leakage.

## Workflows and entry points
- Install dependencies: `pip install -r requirements.txt`.
- Run stages via `python main.py <command>` (see `main.py` subcommands and README examples).
- LLM workflows require optional deps (`torch`, `transformers`, `datasets`); `llm_verification.py` raises clear import errors when missing.

## Examples to follow
- Deliberation and linguistic analysis write JSON reports + figures (`src/analysis/deliberation_metrics.py`, `src/analysis/linguistic_bias.py`).
- Ethics audit produces `results/reports/ethics_audit_report.json` and prints warnings (`src/analysis/ethics_audit.py`).
- Confusion matrix plotting and metrics live in `src/experiments/evaluation.py` and `src/utils/metrics.py`.
