from __future__ import annotations

"""
Two interchangeable classifiers:

- DistilBERT (Transformers), loaded from models/distilbert_twitter/final
- sklearn pipeline (.joblib), loaded from models/

Both expose .predict(texts, return_all_scores=False) -> list[dict].
"""

from functools import lru_cache
from typing import Any, Dict, List, Optional

from .paths import DISTILBERT_DIR, DEFAULT_SKLEARN_PIPELINE
from .data_prep import batch_prep


# ---------- DistilBERT (Transformers) ----------
class DistilBertClassifier:
    def __init__(self, model_dir=None, device: Optional[int] = None):
        # Import inside to avoid hard import errors when user runs only sklearn path
        from transformers import (
            AutoConfig,
            AutoTokenizer,
            AutoModelForSequenceClassification,
            pipeline as hf_pipeline,
        )
        import torch

        self.model_dir = model_dir or DISTILBERT_DIR
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"DistilBERT model not found at {self.model_dir}. "
                f"Please ensure 'models/distilbert_twitter/final' exists."
            )

        self.config = AutoConfig.from_pretrained(self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)

        # Keep a simple integer device id for re-use
        if device is None:
            self.device_id = 0 if torch.cuda.is_available() else -1
        else:
            self.device_id = int(device)

        self.pipe = hf_pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device_id,
            truncation=True,
            return_all_scores=False,
        )

        # id2label mapping if available; otherwise LABEL_{i}
        self.id2label = getattr(self.config, "id2label", None) or {
            i: f"LABEL_{i}" for i in range(getattr(self.config, "num_labels", 2))
        }

    def _map_label(self, raw: str) -> str:
        if raw.startswith("LABEL_"):
            try:
                idx = int(raw.split("_")[-1])
                return self.id2label.get(idx, raw)
            except Exception:
                return raw
        return raw

    def predict(
        self, texts: List[str], return_all_scores: bool = False
    ) -> List[Dict[str, Any]]:
        cleaned = batch_prep(texts, kind="transformer")

        if return_all_scores:
            # Build a separate pipeline that returns all scores; keep self.pipe unchanged
            from transformers import pipeline as hf_pipeline

            pipe_all = hf_pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device_id,
                truncation=True,
                return_all_scores=True,
            )
            outputs = pipe_all(cleaned)
            results: List[Dict[str, Any]] = []
            for out in outputs:
                mapped = [
                    {"label": self._map_label(d["label"]), "score": float(d["score"])}
                    for d in out
                ]
                mapped.sort(key=lambda x: x["score"], reverse=True)
                results.append({"top": mapped[0], "all_scores": mapped})
            return results

        outputs = self.pipe(cleaned)
        return [
            {"label": self._map_label(d["label"]), "score": float(d["score"])}
            for d in outputs
        ]


# ---------- sklearn TF-IDF Logistic Regression (.joblib) ----------
@lru_cache(maxsize=2)
def _load_joblib(path: str):
    import joblib

    return joblib.load(path)


class SklearnPipelineClassifier:
    def __init__(self, pipeline_path=None):
        self.pipeline_path = str(pipeline_path or DEFAULT_SKLEARN_PIPELINE or "")
        if not self.pipeline_path:
            raise FileNotFoundError(
                "No sklearn pipeline found. Expected one of:\n"
                "  models/logreg_tfidf_optuna.joblib\n"
                "  models/logreg_tfidf.joblib"
            )
        self.pipeline = _load_joblib(self.pipeline_path)
        try:
            self.classes_ = list(self.pipeline.classes_)
        except Exception:
            self.classes_ = None

    def predict(
        self, texts: List[str], return_all_scores: bool = False
    ) -> List[Dict[str, Any]]:
        cleaned = batch_prep(texts, kind="sklearn")

        if return_all_scores and hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba(cleaned)
            results: List[Dict[str, Any]] = []
            for row in proba:
                pairs = [
                    {
                        "label": str(self.classes_[i] if self.classes_ else i),
                        "score": float(s),
                    }
                    for i, s in enumerate(row)
                ]
                pairs.sort(key=lambda x: x["score"], reverse=True)
                results.append({"top": pairs[0], "all_scores": pairs})
            return results

        labels = self.pipeline.predict(cleaned)
        scores = None
        if hasattr(self.pipeline, "predict_proba"):
            import numpy as np

            proba = self.pipeline.predict_proba(cleaned)
            scores = np.max(proba, axis=1)

        out: List[Dict[str, Any]] = []
        for i, label in enumerate(labels):
            out.append(
                {
                    "label": str(label),
                    "score": float(scores[i]) if scores is not None else None,
                }
            )
        return out


# ---------- Model loader ----------
def load_model(name: Optional[str] = "distilbert"):
    """
    name in {"distilbert", "sklearn", "logreg", "baseline", "tfidf"}
    Falls back gracefully to whatever exists.
    """
    name = (name or "distilbert").lower()
    if name in {"distilbert", "hf", "transformer", "transformers"}:
        try:
            return DistilBertClassifier()
        except Exception:
            # fall back if the HF model isn't present
            return SklearnPipelineClassifier()
    if name in {"sklearn", "logreg", "baseline", "tfidf"}:
        return SklearnPipelineClassifier()

    # Automatic: try HF first, then sklearn
    try:
        return DistilBertClassifier()
    except Exception:
        return SklearnPipelineClassifier()
