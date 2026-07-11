"""
Inference wrapper for the fine-tuned sentiment model.

Loads the model once and exposes a simple `.predict(text)` API that
returns a label and a calibrated confidence score, so callers (CLI,
Streamlit app, API) all share one source of truth for prediction logic.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)

LABEL_MAP = {0: "Negative", 1: "Positive"}


@dataclass
class Prediction:
    label: str
    confidence: float
    is_confident: bool

    def __str__(self) -> str:
        flag = "" if self.is_confident else " (low confidence)"
        return f"{self.label} ({self.confidence:.2%}){flag}"


class SentimentPredictor:
    def __init__(self, model_dir: Union[str, Path] = config.model_dir):
        model_dir = str(model_dir)
        logger.info(f"Loading model from {model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str) -> Prediction:
        if not text or not text.strip():
            raise ValueError("Input text must be non-empty.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=config.max_seq_length,
        )
        logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=1)

        predicted_class = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0][predicted_class].item())

        return Prediction(
            label=LABEL_MAP[predicted_class],
            confidence=confidence,
            is_confident=confidence >= config.low_confidence_threshold,
        )


if __name__ == "__main__":
    predictor = SentimentPredictor()
    sample = "The product broke after two days, totally disappointed."
    print(predictor.predict(sample))
