# INFACT: A Romanian Institutional Fact-Checking Corpus for Deliberation-Aware NLP

INFACT is a reproducible NLP research pipeline for analyzing Romanian political fact-checking data. The repository includes utilities for dataset preprocessing, exploratory data analysis, baseline and transformer-based claim verification, LLM-based verification runners, and downstream analyses (deliberation metrics, linguistic framing/bias, and ethics audits).

## What’s in this repo

At a high level:

- **`main.py`** — command-line entry point that orchestrates dataset loading, experiments, and analyses.
- **`data/`** — place the TSV dataset(s) here (raw and/or processed).
- **`src/`** — Python package with the pipeline modules (preprocessing, EDA, experiments, LLM runners, analyses, utilities).
- **`scripts/`** — standalone helper scripts (e.g., evaluation utilities).
- **`requirements.txt`** — Python dependencies.
- **`LICENSE`** — project license.

## Repository layout (current)

```
.
├── main.py
├── requirements.txt
├── README.md
├── LICENSE
├── data/
├── scripts/
│   └── evaluate_alignment.py
└── src/
    ├── __init__.py
    ├── analysis/
    │   ├── __init__.py
    │   ├── deliberation_metrics.py
    │   ├── ethics_audit.py
    │   └── linguistic_bias.py
    ├── data_preprocessing/
    │   ├── __init__.py
    │   ├── add_fields.py
    │   ├── balance_infact.py
    │   ├── canonicalize_infact_labels.py
    │   ├── label_mapping.py
    │   ├── load_dataset.py
    │   └── resample.py
    ├── eda/
    │   ├── __init__.py
    │   └── corpus_statistics.py
    ├── experiments/
    │   ├── __init__.py
    │   ├── baseline_verification.py
    │   ├── evaluation.py
    │   ├── llm_verification.py
    │   └── transformer_baselines.py
    ├── llm/
    │   ├── __init__.py
    │   ├── ollama_llama3_1_runner.py
    │   ├── qwen25_7b_infact_runner.py
    │   └── run_qwen25_7b.py
    └── utils/
        ├── __init__.py
        ├── io.py
        ├── metrics.py
        └── text_processing.py
```

Notes:
- This repo also contains editor / metadata directories such as **`.github/`**, **`.vscode/`**, and some **`.DS_Store`** files.

## Dataset

The dataset is expected to be a TSV file with columns such as:

| Column | Description |
|---|---|
| `record_id` | Unique identifier for each claim record |
| `source_url` | URL of the original source |
| `date_verified` | Date the claim was fact-checked |
| `author_claim` | Author or speaker of the claim |
| `source_outlet` | Media outlet or platform where the claim appeared |
| `claim_text` | The full text of the political claim |
| `context` | Contextual information surrounding the claim |
| `verification_scope` | Scope of the fact-checking verification |
| `verification` | Detailed verification narrative |
| `conclusion` | Summary conclusion of the fact-check |
| `domain_claim` | Topic domain of the claim (e.g., health, economy) |
| `verdict_original` | Original verdict label (e.g., True, False, Misleading) |

Place your dataset TSV inside `data/` before running commands.

## Installation

```bash
pip install -r requirements.txt
```

## API keys (.env)

For hosted LLM runs, store secrets in a local `.env` file (gitignored), for example:

```
MISTRAL_API_KEY=your_key_here
HF_API_TOKEN=your_hf_token_here
```

If `python-dotenv` is installed, the runners can load `.env` automatically.

## Usage

`main.py` is the CLI entry point. Commands accept `--data_path` to specify the TSV dataset (defaults to `data/infact_dataset.tsv` if implemented that way in the CLI).

### Run corpus statistics (EDA)

```bash
python main.py stats --data_path data/infact_dataset.tsv
```

### Run baseline verification experiments

```bash
python main.py baseline --data_path data/infact_dataset.tsv
```

### Run LLM verification experiments

```bash
python main.py llm --data_path data/infact_dataset.tsv --model_name bert-base-multilingual-cased
```

### Run transformer baselines

```bash
python main.py transformers --data_path data/infact_dataset.tsv
```

### Run deliberation analysis

```bash
python main.py deliberation --data_path data/infact_dataset.tsv
```

### Run linguistic framing analysis

```bash
python main.py linguistic --data_path data/infact_dataset.tsv
```

### Run ethics/bias audit

```bash
python main.py ethics --data_path data/infact_dataset.tsv
```

### Run local LLM runners

Use module invocation for anything under `src/` to ensure imports resolve:

```bash
python -m src.llm.run_qwen25_7b
python -m src.llm.ollama_llama3_1_runner
```

## Outputs

Depending on which commands you run, outputs are typically written to a local `results/` directory (figures, tables, and reports). If you don’t see `results/` tracked in the repo, it may be generated at runtime and ignored by git.

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**.
