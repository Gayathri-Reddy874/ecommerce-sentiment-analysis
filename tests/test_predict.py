from unittest.mock import MagicMock, patch

import pytest
import torch

from src.predict import Prediction, SentimentPredictor


class TestPrediction:
    def test_str_confident(self):
        pred = Prediction(label="Positive", confidence=0.92, is_confident=True)
        assert "Positive" in str(pred)
        assert "low confidence" not in str(pred)

    def test_str_low_confidence(self):
        pred = Prediction(label="Negative", confidence=0.51, is_confident=False)
        assert "low confidence" in str(pred)


class TestSentimentPredictor:
    @pytest.fixture
    def predictor(self):
        with patch("src.predict.AutoTokenizer") as mock_tok, patch(
            "src.predict.AutoModelForSequenceClassification"
        ) as mock_model_cls:
            fake_encoding = {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
            mock_tokenizer = MagicMock(return_value=fake_encoding)
            mock_tok.from_pretrained.return_value = mock_tokenizer

            mock_model = MagicMock()
            mock_model.eval.return_value = None
            mock_model.return_value = MagicMock(logits=torch.tensor([[0.1, 2.5]]))  # favors class 1
            mock_model_cls.from_pretrained.return_value = mock_model

            yield SentimentPredictor(model_dir="fake/path")

    def test_empty_text_raises(self, predictor):
        with pytest.raises(ValueError):
            predictor.predict("")

    def test_predict_returns_prediction(self, predictor):
        result = predictor.predict("Great product, works perfectly.")
        assert isinstance(result, Prediction)
        assert result.label in {"Positive", "Negative"}
        assert 0.0 <= result.confidence <= 1.0
