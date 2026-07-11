"""Streamlit UI for the e-commerce review sentiment classifier."""
import sys
from pathlib import Path

# Streamlit runs this file directly, which only puts src/ on sys.path,
# not the project root. Add the root so `from src...` imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from src.config import config  # noqa: E402
from src.predict import SentimentPredictor  # noqa: E402

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="🛍️", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def load_predictor() -> SentimentPredictor:
    return SentimentPredictor(model_dir=config.model_dir)


def main() -> None:
    st.title("🛍️ E-commerce Review Sentiment Analyzer")
    st.caption("Fine-tuned transformer model predicting Positive / Negative review sentiment.")

    try:
        predictor = load_predictor()
    except OSError:
        st.error(
            f"No trained model found at `{config.model_dir}`. "
            f"Run `python -m src.train` first to train and save a model."
        )
        return

    user_input = st.text_area(
        "Enter a product review", height=120, placeholder="e.g. The battery life is amazing!"
    )

    if st.button("Analyze", type="primary"):
        if not user_input.strip():
            st.warning("Please enter some text.")
            return

        with st.spinner("Analyzing..."):
            result = predictor.predict(user_input)

        if result.label == "Positive":
            st.success(str(result))
        else:
            st.error(str(result))

        if not result.is_confident:
            st.info(
                f"Confidence is below the {config.low_confidence_threshold:.0%} threshold — "
                f"treat this prediction as uncertain."
            )


if __name__ == "__main__":
    main()
