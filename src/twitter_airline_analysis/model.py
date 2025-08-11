"""
Baseline text‑classification models for Twitter Airline Sentiment.
Only pure functions – side‑effect free – except `save_model`.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def train_val_split(df: pd.DataFrame, *, test_size: float = 0.2):
    """Stratified train/validation split on `airline_sentiment`."""
    return train_test_split(
        df["clean_text"],
        df["airline_sentiment"],
        test_size=test_size,
        stratify=df["airline_sentiment"],
        random_state=RANDOM_STATE,
    )


def build_logreg_pipeline(**logreg_kwargs) -> Pipeline:
    """TF‑IDF ➜ Logistic‑Regression pipeline."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            (
                "clf",
                LogisticRegression(
                    max_iter=200,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    **logreg_kwargs
                ),
            ),
        ]
    )


def evaluate(model: Pipeline, X_val, y_val) -> dict:
    """Return dict with classification metrics & confusion matrix."""
    y_pred = model.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True)
    cm = confusion_matrix(y_val, y_pred)
    return {"report": report, "conf_mat": cm}


def save_model(model: Pipeline, name: str = "tfidf_logreg.pkl") -> Path:
    """Persist fitted model to disk and return its path."""
    out_path = MODEL_DIR / name
    joblib.dump(model, out_path)
    return out_path
