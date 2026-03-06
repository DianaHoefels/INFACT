# Exploratory Data Analysis Report

**Total records:** 10  
**Columns:** 20

## Verdict Distribution
| verdict_original       |   count |   percent |
|:-----------------------|--------:|----------:|
| fals                   |       4 |        40 |
| adevărat               |       2 |        20 |
| parțial adevărat       |       1 |        10 |
| imposibil de verificat |       1 |        10 |
| trunchiat              |       1 |        10 |
| parțial fals           |       1 |        10 |

**Class imbalance ratio (verdict_original):** 4.0

## Epistemic Outcome Distribution
| epistemic_outcome   |   count |   percent |
|:--------------------|--------:|----------:|
| false               |       4 |        40 |
| partial             |       3 |        30 |
| true                |       2 |        20 |
| unverifiable        |       1 |        10 |

**Class imbalance ratio (epistemic_outcome):** 4.0

## Domain Distribution
| domain_claim   |   count |   percent |
|:---------------|--------:|----------:|
| Economie       |       5 |        50 |
| Sanatate       |       2 |        20 |
| Politica       |       1 |        10 |
| Social         |       1 |        10 |
| Educatie       |       1 |        10 |

## Top 10 Authors
| author_claim   |   count |
|:---------------|--------:|
| Ion Popescu    |       3 |
| Maria Ionescu  |       3 |
| Ana Stancu     |       2 |
| Gheorghe Dan   |       2 |

## Top 10 Source Outlets
| source_outlet   |   count |
|:----------------|--------:|
| Digi24          |       3 |
| Pro TV          |       3 |
| Antena3         |       2 |
| Euronews        |       2 |

## Missingness
| column              |   missing_count |   missing_percent |
|:--------------------|----------------:|------------------:|
| verification_binary |               1 |                10 |

## Section Length Statistics
|       |   word_count_claim_text |   word_count_context |   word_count_verification_scope |   word_count_verification |   word_count_conclusion |
|:------|------------------------:|---------------------:|--------------------------------:|--------------------------:|------------------------:|
| count |                    10   |                   10 |                              10 |                      10   |                    10   |
| mean  |                     3.8 |                    2 |                               1 |                       3.5 |                     1.5 |
| std   |                     0.9 |                    0 |                               0 |                       1.4 |                     0.5 |
| min   |                     2   |                    2 |                               1 |                       2   |                     1   |
| 25%   |                     3.2 |                    2 |                               1 |                       3   |                     1   |
| 50%   |                     4   |                    2 |                               1 |                       3   |                     1.5 |
| 75%   |                     4   |                    2 |                               1 |                       4   |                     2   |
| max   |                     5   |                    2 |                               1 |                       7   |                     2   |

## Figures Generated
- `figures/verdict_distribution.png`
- `figures/verification_length_hist.png`
- `figures/domain_distribution.png`
- `figures/top_authors.png`
