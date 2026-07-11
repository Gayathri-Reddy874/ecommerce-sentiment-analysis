"""
Central configuration for the sentiment analysis project.

All tunable parameters are read from environment variables with sane
defaults, so the same code works locally, in CI, and in Docker without
edits. Copy `.env.example` to `.env` and adjust values as needed.
"""
from dataclasses import dataclass
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    # --- Data ---
    raw_data_path: Path = PROJECT_ROOT / "data" / "raw" / "ecommerce_reviews.csv"
    min_review_length: int = 10

    # --- Model ---
    base_model_name: str = os.getenv("BASE_MODEL_NAME", "distilbert-base-uncased")
    model_dir: Path = Path(os.getenv("MODEL_DIR", str(PROJECT_ROOT / "models" / "sentiment_model")))
    num_labels: int = 2
    max_seq_length: int = 128

    # --- Training ---
    num_train_epochs: int = int(os.getenv("NUM_TRAIN_EPOCHS", 4))
    train_batch_size: int = int(os.getenv("TRAIN_BATCH_SIZE", 16))
    eval_batch_size: int = int(os.getenv("EVAL_BATCH_SIZE", 32))
    learning_rate: float = float(os.getenv("LEARNING_RATE", 2e-5))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", 0.01))
    test_size: float = float(os.getenv("TEST_SIZE", 0.2))
    random_seed: int = int(os.getenv("RANDOM_SEED", 42))

    # --- Inference ---
    low_confidence_threshold: float = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", 0.60))

    # --- Logging ---
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


config = Config()
