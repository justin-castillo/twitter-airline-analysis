from twitter_airline_analysis import model
import pandas as pd
from pathlib import Path
import joblib

MODEL_PATH = Path("models/logreg_tfidf.joblib")


def _load_sample(n=500):
    df = pd.read_parquet("data/processed/tweets.parquet").sample(n, random_state=0)
    return df


def test_train_val_split_shapes():
    df = _load_sample()
    X_tr, X_v, y_tr, y_v = model.train_val_split(df, test_size=0.2)
    assert len(X_tr) == len(y_tr)
    assert len(X_v) == len(y_v)
    assert len(X_tr) + len(X_v) == len(df)


def test_pipeline_fits_and_scores():
    df = _load_sample()
    X_tr, X_v, y_tr, y_v = model.train_val_split(df)
    pipe = model.build_logreg_pipeline()
    pipe.fit(X_tr, y_tr)
    metrics = model.evaluate(pipe, X_v, y_v)
    # baseline F1 should be > 0.5 on this sample
    assert metrics["report"]["weighted avg"]["f1-score"] > 0.5


def test_artifact_exists():
    assert MODEL_PATH.exists(), "Baseline model artefact missing"


def test_inference_shape():
    model = joblib.load(MODEL_PATH)
    sample = pd.Series(
        ["Great service but delayed flight"],
        name="clean_text",
    )
    pred = model.predict(sample)
    assert pred.shape == (1,)
