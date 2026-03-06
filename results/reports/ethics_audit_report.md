# Ethics Audit Report

**Total warnings:** 7  
**High severity:** 0  
**Medium severity:** 3  

## Representation Bias

**author_claim**: 2 unique values, imbalance ratio = 1.5, top-5 share = 100.0%, severity = **low**  
Most common: *Ion Popescu* (3 claims)

**domain_claim**: 3 unique values, imbalance ratio = 5.0, top-5 share = 100.0%, severity = **low**  
Most common: *Economie* (5 claims)

**source_outlet**: 2 unique values, imbalance ratio = 1.5, top-5 share = 100.0%, severity = **low**  
Most common: *Digi24* (3 claims)

## Outcome Bias

**author_claim**: mean false rate = 0.4167, std = 0.2887, severity = **medium**

Top entities with highest 'false' classification rate:
  - Ion Popescu: 66.7%
  - Ana Stancu: 50.0%
  - Gheorghe Dan: 50.0%
  - Maria Ionescu: 0.0%

**domain_claim**: mean false rate = 0.28, std = 0.4382, severity = **medium**

Top entities with highest 'false' classification rate:
  - Sanatate: 100.0%
  - Economie: 40.0%
  - Educatie: 0.0%
  - Politica: 0.0%
  - Social: 0.0%

## Linguistic Bias

Hedge rate range across outcomes: 0.0000  
Certainty rate range: 25.0000  
Severity: **medium**

Hedge rate by outcome:
  - false: 0.0000
  - partial: 0.0000
  - true: 0.0000
  - unverifiable: 0.0000

## Temporal Bias

Years covered: 2019–2023  
Missing years: none  
Severity: **low**

## Warnings Summary

| Domain | Sub-key | Severity |
|---|---|---|
| outcome_bias | author_claim | **medium** |
| outcome_bias | domain_claim | **medium** |
| linguistic_bias_check | linguistic_bias_check | **medium** |
| representation_bias | author_claim | **low** |
| representation_bias | domain_claim | **low** |
| representation_bias | source_outlet | **low** |
| temporal_bias | temporal_bias | **low** |

## Mitigation Recommendations

- ℹ️ [MEDIUM] Investigate structural factors driving high 'false' rates for specific actors. Stratify model evaluation by domain and author to detect systematic disparities.
- ℹ️ [MEDIUM] Linguistic markers differ across verdict categories. Consider controlling for framing effects when building classifiers, or use debiasing techniques.
- 💡 [LOW] Consider over-sampling under-represented authors/domains or applying class-weighted models to address representation imbalance.
- 💡 [LOW] Temporal coverage gaps may introduce recency bias. Consider temporal cross-validation and report model performance per time period.
