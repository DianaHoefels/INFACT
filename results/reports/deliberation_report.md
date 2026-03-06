# Deliberation Analysis Report

## Concentration Summary
| dimension     |   unique_values |   total_claims |   gini |   top10_share |
|:--------------|----------------:|---------------:|-------:|--------------:|
| author_claim  |               4 |             10 |   0.1  |             1 |
| domain_claim  |               5 |             10 |   0.36 |             1 |
| source_outlet |               4 |             10 |   0.1  |             1 |

### Interpretation
A Gini coefficient close to 1 indicates high concentration (a small number of actors account for most of the claims). Top-10 share shows the fraction of claims held by the 10 most active entities.

## Yearly Verdict Trends
Data spans 2019–2023.

|   year_verified |   false |   partial |   true |   unverifiable |
|----------------:|--------:|----------:|-------:|---------------:|
|            2019 |       0 |         1 |      0 |              0 |
|            2020 |       2 |         0 |      0 |              0 |
|            2021 |       2 |         1 |      0 |              0 |
|            2022 |       0 |         1 |      2 |              0 |
|            2023 |       0 |         0 |      0 |              1 |

## Verdict Distribution by Domain
_Row-normalised (proportion within each domain)_

| domain_claim   |   false |   partial |   true |   unverifiable |
|:---------------|--------:|----------:|-------:|---------------:|
| Economie       |     0.4 |       0.4 |    0.2 |              0 |
| Educatie       |     0   |       1   |    0   |              0 |
| Politica       |     0   |       0   |    1   |              0 |
| Sanatate       |     1   |       0   |    0   |              0 |
| Social         |     0   |       0   |    0   |              1 |

## Figures
- `figures/yearly_verdict_trends.png`
