"""
resample.py
-----------
Utilities for correcting class imbalance in the INFACT dataset.
Supports random oversampling and undersampling.
"""

import pandas as pd
from sklearn.utils import resample

def oversample_minority(
    df: pd.DataFrame,
    label_col: str = "verdict_normalized",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Randomly oversample minority classes to match the majority class size.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a label column.
    label_col : str
        Name of the label column.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with balanced classes.
    """
    classes = df[label_col].unique()
    max_count = df[label_col].value_counts().max()
    frames = []
    for cls in classes:
        subset = df[df[label_col] == cls]
        frames.append(resample(
            subset,
            replace=True,
            n_samples=max_count,
            random_state=random_state
        ))
    return pd.concat(frames).sample(frac=1, random_state=random_state).reset_index(drop=True)

def undersample_majority(
    df: pd.DataFrame,
    label_col: str = "verdict_normalized",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Randomly undersample majority classes to match the minority class size.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a label column.
    label_col : str
        Name of the label column.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame with balanced classes.
    """
    classes = df[label_col].unique()
    min_count = df[label_col].value_counts().min()
    frames = []
    for cls in classes:
        subset = df[df[label_col] == cls]
        frames.append(resample(
            subset,
            replace=False,
            n_samples=min_count,
            random_state=random_state
        ))
    return pd.concat(frames).sample(frac=1, random_state=random_state).reset_index(drop=True)