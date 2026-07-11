"""
Fine-tune a transformer model for binary review sentiment classification.

Usage:
    python -m src.train
    python -m src.train --epochs 3 --batch-size 8 --model distilbert-base-uncased
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.config import config
from src.data_processing import prepare_dataset
from src.dataset import ReviewDataset
from src.logger import get_logger

logger = get_logger(__name__)


class WeightedTrainer(Trainer):
    """Trainer that applies class weights to the loss, so a majority class
    (e.g. mostly-positive real-world reviews) doesn't dominate training."""

    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the review sentiment model.")
    parser.add_argument("--model", default=config.base_model_name)
    parser.add_argument("--epochs", type=int, default=config.num_train_epochs)
    parser.add_argument("--batch-size", type=int, default=config.train_batch_size)
    parser.add_argument("--learning-rate", type=float, default=config.learning_rate)
    parser.add_argument("--output-dir", default=str(config.model_dir))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_texts, val_texts, train_labels, val_labels = prepare_dataset()

    logger.info(f"Loading tokenizer and base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=config.num_labels,
        problem_type="single_label_classification",
    )

    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=config.max_seq_length
    )
    val_encodings = tokenizer(
        val_texts, truncation=True, padding=True, max_length=config.max_seq_length
    )

    train_dataset = ReviewDataset(train_encodings, train_labels)
    val_dataset = ReviewDataset(val_encodings, val_labels)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1]), y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    logger.info(f"Class weights (negative, positive): {class_weights.tolist()}")

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=config.weight_decay,
        logging_dir="./logs",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=config.random_seed,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights,
    )

    logger.info("Starting training...")
    trainer.train()

    metrics = trainer.evaluate()
    logger.info(f"Final validation metrics: {metrics}")

    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
