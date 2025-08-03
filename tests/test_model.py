from pathlib import Path
import joblib                               # required for .load()
import pandas as pd

from twitter_airline_analysis import model

# ---------------------------------------------------------
# Paths
MODEL_PATH = Path("models/logreg_tfidf.joblib")

# ---------------------------------------------------------
def _load_sample(n: int = 500):
    """Grab a deterministic sample of the processed parquet."""
    df = pd.read_parquet("data/processed/tweets.parquet").sample(
        n, random_state=0
    )
    return df


# ---------- shape / split tests ---------------------------------------------
def test_train_val_split_shapes():
    df = _load_sample()
    X_tr, X_v, y_tr, y_v = model.train_val_split(df, test_size=0.2)

    assert len(X_tr) == len(y_tr)
    assert len(X_v) == len(y_v)
    assert len(X_tr) + len(X_v) == len(df)


# ---------- pipeline fit / score test ---------------------------------------
def test_pipeline_fits_and_scores():
    df = _load_sample()
    X_tr, X_v, y_tr, y_v = model.train_val_split(df, test_size=0.2)
    pipe = model.build_logreg_pipeline()
    pipe.fit(X_tr, y_tr)

    metrics = model.evaluate(pipe, X_v, y_v)
    # Very small sampled set, but F1 should be > 0.5
    assert metrics["report"]["weighted avg"]["f1-score"] > 0.5


# ---------- artefact + inference tests --------------------------------------
def test_artifact_exists():
    assert MODEL_PATH.exists(), "Baseline model artefact missing"


def test_inference_shape():
    clf = joblib.load(MODEL_PATH)
    sample = pd.Series(
        ["Great service but delayed flight"], name="clean_text"
    )
    pred = clf.predict(sample)
    assert pred.shape == (1,)
