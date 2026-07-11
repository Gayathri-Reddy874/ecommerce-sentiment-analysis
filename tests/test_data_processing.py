import pandas as pd
import pytest

from src.data_processing import _rating_to_label, clean_and_label


class TestRatingToLabel:
    def test_high_rating_is_positive(self):
        assert _rating_to_label(4) == 1
        assert _rating_to_label(5) == 1

    def test_low_rating_is_negative(self):
        assert _rating_to_label(1) == 0
        assert _rating_to_label(2) == 0

    def test_neutral_rating_is_none(self):
        assert _rating_to_label(3) is None


class TestCleanAndLabel:
    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "review_id": [1, 2, 3, 4, 5, 6],
                "product": ["A"] * 6,
                "rating": [5, 1, 3, 4, 4, 2],
                "review_text": [
                    "This product is absolutely wonderful and useful.",
                    "Terrible quality, broke immediately after use.",
                    "It's okay, does the job I guess.",
                    "Too short",  # < 10 chars, should be dropped
                    "This product is absolutely wonderful and useful.",  # duplicate
                    "Bad",  # too short
                ],
            }
        )

    def test_drops_neutral_ratings(self, sample_df):
        result = clean_and_label(sample_df)
        assert 3 not in result["rating"].values

    def test_drops_short_reviews(self, sample_df):
        result = clean_and_label(sample_df)
        assert all(result["review_text"].str.len() >= 10)

    def test_keeps_duplicate_text_rows(self, sample_df):
        # Duplicate review text is common in this dataset (many rows share a
        # handful of templated phrases with different ratings). We keep every
        # row rather than silently dropping "duplicates", since collapsing
        # them would discard real, differently-labeled training examples.
        result = clean_and_label(sample_df)
        assert result["review_text"].duplicated().sum() == 1

    def test_label_column_is_binary(self, sample_df):
        result = clean_and_label(sample_df)
        assert set(result["label"].unique()).issubset({0, 1})
