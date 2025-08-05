"""CLI inference utility for the airline‑sentiment model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "logreg_tfidf.joblib"
_PIPE = joblib.load(MODEL_PATH)         # eager load; model ≈1 MB

def predict(text: str) -> str:
    """Return the sentiment class for a single text."""
    return _PIPE.predict([text])[0]

def main() -> None:                     # pragma: no cover
    parser = argparse.ArgumentParser(description="Predict tweet sentiment.")
    parser.add_argument("text", nargs="+", help="Text to classify.")
    args = parser.parse_args()
    print(predict(" ".join(args.text)))

if __name__ == "__main__":
    main()
