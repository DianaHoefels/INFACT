# INFACT: A Romanian Institutional Fact-Checking Corpus for Deliberation-Aware NLP

INFACT is a reproducible NLP research pipeline for analyzing a fact-checking dataset of Romanian
political claims. It supports label engineering, exploratory data analysis, baseline and LLM-based
claim verification, deliberation-aware discourse analysis, linguistic framing analysis, and
bias/ethics auditing.

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

## Repository Structure

```
infact/
├── README.md
├── requirements.txt
├── main.py
│
├── data/                        # Place dataset TSV here
│
├── src/
│   ├── data/
│   │   ├── load_dataset.py      # Data loading and validation
│   │   └── label_mapping.py     # Label engineering for epistemic outcomes
│   │
│   ├── eda/
│   │   └── corpus_statistics.py # Exploratory data analysis
│   │
│   ├── experiments/
│   │   ├── baseline_verification.py  # Baseline claim verification (TF-IDF + ML)
│   │   ├── llm_verification.py       # LLM-based claim verification
│   │   └── evaluation.py             # Evaluation utilities
│   │
│   ├── analysis/
│   │   ├── deliberation_metrics.py   # Deliberation-aware discourse analysis
│   │   ├── linguistic_bias.py        # Linguistic framing analysis
│   │   └── ethics_audit.py           # Bias and ethics auditing
│   │
│   └── utils/
│       ├── io.py                # I/O utilities
│       ├── metrics.py           # Evaluation metrics
│       └── text_processing.py   # Text pre-processing helpers
│
└── results/
    ├── tables/    # Saved CSV/LaTeX tables
    ├── figures/   # Saved plots
    └── reports/   # Saved text/JSON reports
```

## Installation

```bash
pip install -r requirements.txt
```

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

## Expected Outputs

After running experiments, the `results/` directory will contain:

- `results/figures/` — distribution plots, confusion matrices, framing heatmaps
- `results/tables/` — classification reports, metric tables in CSV format
- `results/reports/` — JSON reports for ethics audit and deliberation metrics

## Citation

If you use INFACT in your research, please cite the dataset and this repository accordingly.

## License

See [LICENSE](LICENSE) for details.
