"""Pytest suite for the baseline TF‑IDF + logistic‑regression pipeline."""

from pathlib import Path

import joblib
import pandas as pd

from twitter_airline_analysis import model

# --------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------- #
MODEL_PATH = Path("models/logreg_tfidf.joblib")


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _load_sample(n: int = 500) -> pd.DataFrame:
    """Return a deterministic sample from the processed tweet parquet."""
    return (
        pd.read_parquet("data/processed/tweets.parquet")
        .sample(n, random_state=0)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #
def test_train_val_split_shapes() -> None:
    df = _load_sample()
    X_tr, X_v, y_tr, y_v = model.train_val_split(df, test_size=0.2)

    assert len(X_tr) == len(y_tr)
    assert len(X_v) == len(y_v)
    assert len(X_tr) + len(X_v) == len(df)


def test_pipeline_fits_and_scores() -> None:
    df = _load_sample()
    X_tr, X_v, y_tr, y_v = model.train_val_split(df, test_size=0.2)

    pipe = model.build_logreg_pipeline()
    pipe.fit(X_tr, y_tr)
    metrics = model.evaluate(pipe, X_v, y_v)

    # Very small sample; expect F1 comfortably above random guessing
    assert metrics["report"]["weighted avg"]["f1-score"] > 0.50


def test_artifact_exists() -> None:
    assert MODEL_PATH.exists(), "Baseline model artefact missing"


def test_inference_shape() -> None:
    clf = joblib.load(MODEL_PATH)
    sample = pd.Series(["Great service but delayed flight"], name="clean_text")
    pred = clf.predict(sample)

    assert pred.shape == (1,)
