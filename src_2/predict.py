"""
Thin, cached prediction API shared by Streamlit and FastAPI.
"""
from functools import lru_cache
from typing import Any, Dict, List, Union

from .model import load_model

@lru_cache(maxsize=8)
def _cached_loader(name: str):
    return load_model(name)

def predict_texts(
    texts: Union[str, List[str]],
    model_name: str = "distilbert",
    return_all_scores: bool = False
) -> List[Dict[str, Any]]:
    if isinstance(texts, str):
        texts = [texts]
    model = _cached_loader((model_name or "distilbert").lower())
    return model.predict(texts, return_all_scores=return_all_scores)
