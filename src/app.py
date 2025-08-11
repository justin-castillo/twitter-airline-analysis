"""FastAPI wrapper exposing /predict endpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "logreg_tfidf.joblib"
PIPE = joblib.load(MODEL_PATH)


class InferenceRequest(BaseModel):
    text: str = Field(..., example="My flight was delayed 3 hours")


class InferenceResponse(BaseModel):
    label: Literal["negative", "neutral", "positive"]


app = FastAPI(
    title="Airline Sentiment Inference API",
    version="0.1.0",
    summary="Lightweight TF-IDF + LogReg sentiment classifier",
)


@app.post("/predict", response_model=InferenceResponse)
def predict(req: InferenceRequest) -> InferenceResponse:
    """Return the sentiment label for the supplied text."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text must be non-empty.")
    label = PIPE.predict([req.text])[0]
    return InferenceResponse(label=label)
