# Data

This directory contains the INFACT dataset splits.

## Dataset Statistics

| Split | # Claims | # TRUE | # FALSE | # PARTIALLY_TRUE | # UNVERIFIABLE |
|-------|----------|--------|---------|-----------------|----------------|
| Train | –        | –      | –       | –               | –              |
| Dev   | –        | –      | –       | –               | –              |
| Test  | –        | –      | –       | –               | –              |

> **Note:** The dataset files are not stored in this repository due to size and
> licensing constraints. Please follow the download instructions in the root
> `README.md` to obtain the data.

## File Format

Each split is stored as a JSON Lines (`.jsonl`) file. Every line is a valid JSON
object representing one instance with the following fields:

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

## Field Descriptions

| Field          | Type             | Description                                               |
|----------------|------------------|-----------------------------------------------------------|
| `id`           | `string`         | Unique identifier for the instance                        |
| `claim`        | `string`         | The claim to be verified (in Romanian)                    |
| `label`        | `string`         | Verdict: `TRUE`, `FALSE`, `PARTIALLY_TRUE`, `UNVERIFIABLE` |
| `evidence`     | `list[object]`   | List of evidence pieces supporting the verdict            |
| `claim_source` | `string`         | Platform/outlet where the claim was published             |
| `claim_date`   | `string`         | ISO 8601 date when the claim was made                     |
| `domain`       | `string`         | Topical domain (e.g., health, politics, economy)          |
| `language`     | `string`         | Language code (`ro` for Romanian)                         |

### Evidence Object

| Field    | Type     | Description                              |
|----------|----------|------------------------------------------|
| `source` | `string` | Name of the institutional source         |
| `url`    | `string` | URL of the original document             |
| `date`   | `string` | ISO 8601 date of the evidence document   |
| `text`   | `string` | Relevant excerpt from the evidence       |

## Label Distribution

INFACT uses a four-way verdict taxonomy:

- **TRUE** – The claim is fully supported by evidence.
- **FALSE** – The claim is contradicted by evidence.
- **PARTIALLY_TRUE** – The claim is partially supported/contradicted.
- **UNVERIFIABLE** – Insufficient evidence to determine the verdict.

## Domains

Claims are drawn from the following institutional domains:

- `health` – Healthcare, medicine, public health policy
- `politics` – Government decisions, electoral claims
- `economy` – Economic statistics, financial data
- `environment` – Climate, environmental policy
- `education` – Education policy and statistics
- `other` – Miscellaneous institutional claims
