"""
Data loading, cleaning and label generation for the review dataset.

The raw CSV has columns: review_id, product, rating, review_text.
Ratings of 4-5 are treated as positive (1), 1-2 as negative (0), and
rating 3 (neutral) is dropped since it doesn't cleanly belong to
either class for a binary classifier.
"""
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(path: Path = config.raw_data_path) -> pd.DataFrame:
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    required_cols = {"review_text", "rating"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df


def _rating_to_label(rating: int) -> int | None:
    if rating >= 4:
        return 1
    if rating <= 2:
        return 0
    return None  # neutral rating, excluded from training


def clean_and_label(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)

    df = df.copy()
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"].notna()]
    df = df[df["review_text"].str.len() >= config.min_review_length]

    n_unique = df["review_text"].nunique()
    if n_unique < 50:
        logger.warning(
            f"Only {n_unique} unique review texts found in {len(df)} rows. "
            f"This dataset appears to be templated/synthetic and is not diverse "
            f"enough to train a production-grade model — see README 'Dataset Notes'."
        )

    df["label"] = df["rating"].apply(_rating_to_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    after = len(df)
    logger.info(
        f"Cleaned dataset: {before} -> {after} rows "
        f"({before - after} removed as too short, duplicate, or neutral rating)"
    )
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
    return df.reset_index(drop=True)


def train_val_split(df: pd.DataFrame) -> Tuple[list, list, list, list]:
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["review_text"].tolist(),
        df["label"].tolist(),
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=df["label"].tolist(),
    )
    logger.info(f"Train size: {len(train_texts)} | Validation size: {len(val_texts)}")
    return train_texts, val_texts, train_labels, val_labels


def prepare_dataset() -> Tuple[list, list, list, list]:
    """Convenience entry point: load -> clean -> split."""
    df = load_raw_data()
    df = clean_and_label(df)
    return train_val_split(df)
