# INFACT: A Romanian Institutional Fact-Checking Corpus

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

INFACT is a benchmark dataset for automated fact-checking of claims from Romanian
institutional sources. The corpus pairs naturally occurring claims with structured
evidence from authoritative Romanian institutions and human-annotated verdicts,
enabling research on multilingual and low-resource fact verification.

---

## Table of Contents

- [Overview](#overview)
- [Dataset Statistics](#dataset-statistics)
- [Data Format](#data-format)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Download the Data](#download-the-data)
  - [Loading the Data](#loading-the-data)
- [Evaluation](#evaluation)
- [Baselines](#baselines)
- [Leaderboard](#leaderboard)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Overview

Fact-checking research has largely focused on English-language resources. INFACT
addresses this gap by providing a fact-checking benchmark grounded in the Romanian
institutional domain. Claims are collected from social media and news outlets and
verified against official documents, press releases, and statistical reports
published by Romanian governmental and academic institutions.

**Key features:**

- ✅ Naturally occurring claims in **Romanian**
- 🏛️ Evidence from **institutional sources** (government, academia, NGOs)
- 🏷️ Four-way verdict taxonomy: `TRUE`, `FALSE`, `PARTIALLY_TRUE`, `UNVERIFIABLE`
- 📋 Structured evidence with source URLs and publication dates
- 🗂️ Topical domains: health, politics, economy, environment, education

---

## Dataset Statistics

| Split | # Claims | TRUE | FALSE | PARTIALLY_TRUE | UNVERIFIABLE |
|-------|----------|------|-------|----------------|--------------|
| Train | –        | –    | –     | –              | –            |
| Dev   | –        | –    | –     | –              | –            |
| Test  | –        | –    | –     | –              | –            |
| **Total** | **–** | **–** | **–** | **–**     | **–**        |

> Dataset statistics will be updated upon public release.

---

## Data Format

Each split is stored as a [JSON Lines](https://jsonlines.org/) (`.jsonl`) file.
Every line is a self-contained JSON object:

```json
{
  "id": "infact-train-00001",
  "claim": "Guvernul a alocat 500 de milioane de lei pentru construirea de spitale.",
  "label": "FALSE",
  "evidence": [
    {
      "source": "Ministerul Sănătății",
      "url": "https://example.ro/comunicat",
      "date": "2023-05-10",
      "text": "Suma alocată pentru construcția de spitale este de 50 de milioane de lei."
    }
  ],
  "claim_source": "Facebook",
  "claim_date": "2023-05-12",
  "domain": "health",
  "language": "ro"
}
```

See [`data/README.md`](data/README.md) for a full description of all fields.

---

## Getting Started

### Installation

```bash
git clone https://github.com/DianaHoefels/INFACT.git
cd INFACT
pip install -r requirements.txt
# (optional) install the infact package in editable mode
pip install -e .
```

### Download the Data

```bash
python scripts/download_data.py --data-dir data/
```

> **Note:** The dataset will be made publicly available upon paper acceptance.
> Contact the authors for early access.

### Loading the Data

```python
import sys
sys.path.insert(0, "src")

from data_utils import load_split, dataset_statistics

train = load_split("train")
dev   = load_split("dev")
test  = load_split("test")

print(dataset_statistics(train))
```

---

## Evaluation

Evaluate your predictions against the gold test labels:

```bash
python src/evaluate.py \
    --gold data/test.jsonl \
    --predictions my_predictions.jsonl \
    --output-json results.json
```

The script reports **accuracy** and **macro-averaged F1** along with per-class
precision, recall, and F1 for each verdict category.

Prediction files must be JSON Lines files where each object contains:
- `"id"` – the instance ID from the gold file
- `"label"` – the predicted verdict

---

## Baselines

| Model | Accuracy | Macro-F1 |
|-------|----------|----------|
| Majority class | – | – |
| TF-IDF + Logistic Regression | – | – |
| Romanian BERT ([dumitrescuv/bert-base-romanian-uncased-v1](https://huggingface.co/dumitrescuv/bert-base-romanian-uncased-v1)) | – | – |
| mBERT | – | – |
| XLM-RoBERTa | – | – |

> Baseline results will be reported in the accompanying paper.

---

## Leaderboard

We welcome community submissions. To add your results to the leaderboard, please
open a pull request editing the table above and include a reference to your
system description paper or technical report.

---

## Citation

If you use INFACT in your research, please cite:

```bibtex
@inproceedings{hoefels2024infact,
  title     = {{INFACT}: A Romanian Institutional Fact-Checking Corpus},
  author    = {Hoefels, Diana},
  booktitle = {Proceedings of the Annual Meeting of the Association for Computational Linguistics},
  year      = {2024},
  url       = {https://github.com/DianaHoefels/INFACT}
}
```

---

## License

This project is released under the [Apache License 2.0](LICENSE).

The dataset annotations are released under the same license. Evidence texts
are excerpts from publicly available institutional documents; please refer to
the original sources for their respective licences.

---

## Contact

- **Diana Hoefels** – maintainer
- Open an [issue](https://github.com/DianaHoefels/INFACT/issues) for bug reports
  and feature requests.
- Send an email for questions about data access or collaboration.
