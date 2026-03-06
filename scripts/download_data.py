#!/usr/bin/env python3
"""Download the INFACT dataset splits.

This script downloads the canonical train / dev / test splits and saves
them to the ``data/`` directory.  Update the ``DATASET_URL`` variable below
to point to the actual hosting location once the data is publicly released.

Usage
-----
::

    python scripts/download_data.py [--data-dir data/]
"""

import argparse
import sys
import urllib.request
from pathlib import Path

# TODO: Replace these placeholder URLs with the actual download links once
# the dataset is publicly released.
SPLIT_URLS: dict[str, str] = {
    "train": "https://example.com/infact/train.jsonl",
    "dev": "https://example.com/infact/dev.jsonl",
    "test": "https://example.com/infact/test.jsonl",
}


def download_file(url: str, dest: Path) -> None:
    """Download *url* to *dest*, printing progress."""
    print(f"  Downloading {url}")
    print(f"       -> {dest}")
    urllib.request.urlretrieve(url, dest)  # noqa: S310


def main(argv: list[str] | None = None) -> int:
    """Download the INFACT dataset splits."""
    parser = argparse.ArgumentParser(
        description="Download INFACT dataset splits.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Directory where the splits will be saved (default: data/).",
    )
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading INFACT splits to: {data_dir}\n")

    for split, url in SPLIT_URLS.items():
        dest = data_dir / f"{split}.jsonl"
        if dest.exists():
            print(f"  [{split}] already exists – skipping.")
            continue
        try:
            download_file(url, dest)
            print(f"  [{split}] OK\n")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{split}] FAILED: {exc}", file=sys.stderr)
            return 1

    print("Done.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
