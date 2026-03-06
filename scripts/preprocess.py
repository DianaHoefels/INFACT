#!/usr/bin/env python3
"""Convert raw fact-checking data into the INFACT JSON Lines format.

This script reads raw data (e.g. scraped or annotated spreadsheets) and
writes standardised ``.jsonl`` files for each split.

Usage
-----
::

    python scripts/preprocess.py \\
        --input-dir raw_data/ \\
        --output-dir data/ \\
        --split train

Output
------
One ``.jsonl`` file per split is written to ``--output-dir``.  Each line is a
JSON object conforming to the INFACT schema described in ``data/README.md``.
"""

import argparse
import json
import sys
import uuid
from pathlib import Path


def make_instance(
    claim: str,
    label: str,
    evidence: list[dict],
    *,
    claim_source: str = "",
    claim_date: str = "",
    domain: str = "other",
    language: str = "ro",
    instance_id: str | None = None,
) -> dict:
    """Create an INFACT instance dictionary.

    Args:
        claim: The claim text (in Romanian).
        label: Verdict label – one of ``TRUE``, ``FALSE``,
            ``PARTIALLY_TRUE``, ``UNVERIFIABLE``.
        evidence: List of evidence objects.
        claim_source: Platform / outlet where the claim appeared.
        claim_date: ISO 8601 date the claim was published.
        domain: Topical domain of the claim.
        language: BCP-47 language tag (default: ``"ro"``).
        instance_id: Optional unique identifier. A UUID is generated when
            omitted.

    Returns:
        An instance dict conforming to the INFACT schema.
    """
    valid_labels = {"TRUE", "FALSE", "PARTIALLY_TRUE", "UNVERIFIABLE"}
    if label not in valid_labels:
        raise ValueError(f"Invalid label '{label}'. Must be one of {valid_labels}.")

    return {
        "id": instance_id or str(uuid.uuid4()),
        "claim": claim,
        "label": label,
        "evidence": evidence,
        "claim_source": claim_source,
        "claim_date": claim_date,
        "domain": domain,
        "language": language,
    }


def write_jsonl(instances: list[dict], path: Path) -> None:
    """Write a list of instances to a JSON Lines file.

    Args:
        instances: List of INFACT instance dicts.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for inst in instances:
            fh.write(json.dumps(inst, ensure_ascii=False) + "\n")
    print(f"Wrote {len(instances)} instances to {path}")


def main(argv: list[str] | None = None) -> int:
    """Entry point for the preprocessing script."""
    parser = argparse.ArgumentParser(
        description="Convert raw data into the INFACT JSONL format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing raw input files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Directory where processed ``.jsonl`` files will be written (default: data/).",
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        required=True,
        help="Which dataset split to produce.",
    )

    args = parser.parse_args(argv)

    if not args.input_dir.exists():
        print(f"Error: input directory does not exist: {args.input_dir}", file=sys.stderr)
        return 1

    # TODO: Replace the stub below with actual parsing logic for your raw data.
    print(
        f"Warning: no raw-data parser is implemented yet.\n"
        f"Add parsing logic in {__file__} to populate 'instances'.",
        file=sys.stderr,
    )
    instances: list[dict] = []

    output_path = args.output_dir / f"{args.split}.jsonl"
    write_jsonl(instances, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
