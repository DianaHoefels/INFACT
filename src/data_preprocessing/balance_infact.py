"""
balance_infact.py
-----------------
Script to create a balanced version of the INFACT dataset using oversampling.
"""

import pandas as pd
from src.data_preprocessing.load_dataset import load_infact, validate_dataset
from src.data_preprocessing.label_mapping import apply_label_mapping
from src.data_preprocessing.resample import oversample_minority

if __name__ == "__main__":
    # Load and validate
    df = load_infact("data/infact_dataset.tsv")
    validate_dataset(df)
    df = apply_label_mapping(df)

    # Balance the dataset
    balanced_df = oversample_minority(df)

    # Save to new file
    balanced_df.to_csv("data/infact_dataset_balanced.tsv", sep="\t", index=False)
    print("Balanced dataset saved to data/infact_dataset_balanced.tsv")