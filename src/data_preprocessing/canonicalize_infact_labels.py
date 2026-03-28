from pathlib import Path

import pandas as pd

LABEL_CANONICAL_MAP = {
    "TRUE": "True",
    "True": "True",
    "true": "True",
    "FALSE": "False",
    "False": "False",
    "false": "False",
    "MIXED": "Mixed",
    "Mixed": "Mixed",
    "mixed": "Mixed",
    "MOSTLY TRUE": "Mostly True",
    "Mostly True": "Mostly True",
    "mostly true": "Mostly True",
    "MOSTLY FALSE": "Mostly False",
    "Mostly False": "Mostly False",
    "mostly false": "Mostly False",
    "UNVERIFIABLE": "Unverifiable",
    "Unverifiable": "Unverifiable",
    "unverifiable": "Unverifiable",
}

FILE_PAIRS = [
    ("data/infact.tsv", "data/infact_canonical.tsv"),
    ("data/infact_dataset_processed.tsv", "data/infact_dataset_processed_canonical.tsv"),
]


def canonicalize_verdict_label(x: str) -> str:
    x = str(x).strip()
    if x in LABEL_CANONICAL_MAP:
        return LABEL_CANONICAL_MAP[x]
    raise ValueError(f"Unexpected verdict_normalized label: {x}")


def main() -> None:
    for input_path, output_path in FILE_PAIRS:
        if not Path(input_path).exists():
            print(f"Skipping missing file: {input_path}")
            continue
        df = pd.read_csv(input_path, sep="\t")
        df["verdict_normalized"] = df["verdict_normalized"].apply(
            canonicalize_verdict_label
        )
        df.to_csv(output_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
