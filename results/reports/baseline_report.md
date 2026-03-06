# Baseline Classification Report

## Experiment A: `claim_text`

| Metric | Score |
|---|---|
| Macro F1 | 0.3333 |
| Weighted F1 | 0.3333 |
| Accuracy | 0.5000 |
| Train samples | 4 |
| Test samples | 2 |

### Per-class Results
| class        |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| false        |        0.5  |      1   |   0.666667 |       1   |
| partial      |        0    |      0   |   0        |       1   |
| accuracy     |        0.5  |      0.5 |   0.5      |       0.5 |
| macro avg    |        0.25 |      0.5 |   0.333333 |       2   |
| weighted avg |        0.25 |      0.5 |   0.333333 |       2   |

## Experiment B: `claim_text` + `context`

| Metric | Score |
|---|---|
| Macro F1 | 0.3333 |
| Weighted F1 | 0.3333 |
| Accuracy | 0.5000 |
| Train samples | 4 |
| Test samples | 2 |

### Per-class Results
| class        |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| false        |        0.5  |      1   |   0.666667 |       1   |
| partial      |        0    |      0   |   0        |       1   |
| accuracy     |        0.5  |      0.5 |   0.5      |       0.5 |
| macro avg    |        0.25 |      0.5 |   0.333333 |       2   |
| weighted avg |        0.25 |      0.5 |   0.333333 |       2   |

## Experiment C: `claim_text` + `context` + `verification_scope`

| Metric | Score |
|---|---|
| Macro F1 | 1.0000 |
| Weighted F1 | 1.0000 |
| Accuracy | 1.0000 |
| Train samples | 4 |
| Test samples | 2 |

### Per-class Results
| class        |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| false        |           1 |        1 |          1 |         1 |
| partial      |           1 |        1 |          1 |         1 |
| accuracy     |           1 |        1 |          1 |         1 |
| macro avg    |           1 |        1 |          1 |         2 |
| weighted avg |           1 |        1 |          1 |         2 |

## Summary Table
| experiment   | features                                  |   macro_f1 |   weighted_f1 |   accuracy |   n_train |   n_test |
|:-------------|:------------------------------------------|-----------:|--------------:|-----------:|----------:|---------:|
| A            | claim_text                                |     0.3333 |        0.3333 |        0.5 |         4 |        2 |
| B            | claim_text + context                      |     0.3333 |        0.3333 |        0.5 |         4 |        2 |
| C            | claim_text + context + verification_scope |     1      |        1      |        1   |         4 |        2 |
