"""
FastAPI app (programmatic access).
Run: uvicorn src_2.app:app --reload --host 0.0.0.0 --port 8000
"""

# Ensure project root is on sys.path when run various ways (Streamlit/Uvicorn/VS Code)
import sys
from pathlib import Path

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from typing import List, Optional

try:
    # Python 3.8+ has typing.Literal
    from typing import Literal
except ImportError:  # pragma: no cover
    from typing_extensions import Literal  # type: ignore

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src_2.predict import predict_texts

app = FastAPI(title="Twitter Airline Sentiment (src_2)", version="1.0.0")

class PredictRequest(BaseModel):
    text: Optional[str] = Field(None, description="Single text")
    texts: Optional[List[str]] = Field(None, description="Multiple texts")
    model_name: Optional[Literal["distilbert", "sklearn", "logreg", "baseline", "tfidf"]] = "distilbert"
    return_all_scores: Optional[bool] = False

class PredictResponse(BaseModel):
    predictions: List[dict]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    items = req.texts or ([req.text] if req.text is not None else [])
    if not items:
        return {"predictions": []}
    preds = predict_texts(items, model_name=req.model_name, return_all_scores=req.return_all_scores)
    return {"predictions": preds}
