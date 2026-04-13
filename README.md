# INFACT: A Romanian Institutional Fact-Checking Corpus for Deliberation-Aware NLP

INFACT is a comprehensive, reproducible NLP research pipeline for analyzing Romanian political fact-checking data. Its main goals are to enable robust claim verification, deliberation-aware discourse analysis, and institutional-reasoning–aligned explainability.

It supports label engineering, exploratory data analysis, baseline and transformer-based claim verification, LLM-based claim verification, deliberation-aware discourse analysis, linguistic framing analysis, and bias/ethics auditing.

## Paper / publication status

This project was **accepted** as a paper to be published at **LREC-COLING 2026**, at the **2nd Workshop on Language-driven Deliberation Technology (DELITE)**.

**Accepted reference (details pending / may change):**

Diana Constantina Hoefels. 2026. *InFACT: Benchmarking LLM Explanations Against Institutional Reasoning for Deliberation-Aware Fact-Checking* (accepted at 2nd Workshop on Language-driven Deliberation Technology (DELITE) @ LREC-COLING 2026).

## Citation (NOT OFFICIAL YET)

> This citation is **provisional** and **not final/official** yet.

```bibtex
@InProceedings{hoefels:2022:LREC,
  author    = {Hoefels, Diana Constantina},
  title     = {InFACT: Benchmarking LLM Explanations Against Institutional Reasoning for
Deliberation-Aware Fact-Checking},
  booktitle      = {Proceedings of the Language Resources and Evaluation Conference},
  month          = {May},
  year           = {2026},
  address        = {Palma de Mallorca, Spain},
  publisher      = {European Language Resources Association},
  pages     = {TBA},
  abstract  = {Explainability in deliberation-support NLP is usually evaluated through post-hoc rationales or model-internal attribution methods, and only rarely against explicit institutional reasoning procedures. We introduce \InFACT, a Romanian corpus of professional fact-checking reports that preserves the workflow of editorial epistemic arbitration, namely claim articulation, contextualisation, verification scope, evidence-based verification narrative, and calibrated conclusion. \InFACT\ contains 789 raw reports from \textit{factual.ro} and a processed benchmark release of 788 instances after removal of a singleton non-standard verdict label. Beyond six-way verdict prediction, we position \InFACT\ as a benchmark for LLM explanation alignment, where models must generate short explanations that can be compared directly to gold institutional reasoning. We evaluate \InFACT\ primarily with instruction-tuned LLMs, reporting full-corpus experiments for open-weight models and a matched pilot comparison with GPT-4 Turbo. The resulting evidence shows that verdict prediction and institutional explanation alignment are not the same capability: models that improve verdict accuracy do not necessarily preserve institutional calibration or produce explanations that align with professional verification narratives. These results support the central claim of the paper, namely that \InFACT\ measures not only whether a model reaches a verdict, but also whether it does so in a manner that resembles documented public reasoning.},
  url       = {TBA}
}
```

## Dataset

The dataset is a TSV file with the following columns:

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

Place the dataset TSV file inside the `data/` directory before running experiments.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── main.py
├── data/                        
├── scripts/
│   └── evaluate_alignment.py
└── src/
    ├── __init__.py
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
    ├���─ analysis/
    │   ├── __init__.py
    │   ├── deliberation_metrics.py
    │   ├── linguistic_bias.py
    │   ├── ethics_audit.py
    │   └── deliberation_metrics.py
    └── utils/
        ├── __init__.py
        ├── io.py
        ├── metrics.py
        └── text_processing.py
```

## Installation

```bash
pip install -r requirements.txt
```

### API keys (.env)

For hosted LLM runs, store secrets in a local `.env` file (gitignored):

```
MISTRAL_API_KEY=your_key_here
HF_API_TOKEN=your_hf_token_here
```

The runner will load `.env` automatically if `python-dotenv` is installed.

## Usage

`main.py` is the CLI entry point. All commands accept `--data_path` to specify the TSV dataset
file (defaults to `data/infact_dataset.tsv`).

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

### Run transformer baselines (XLM-R + Romanian BERT)

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

### Run full pipeline

```bash
python main.py all --data_path data/infact_dataset.tsv
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
