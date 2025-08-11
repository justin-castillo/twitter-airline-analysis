"""
Lightweight preprocessing.

- For Transformers: keep it minimal (tokenizer handles casing/punct).
- For sklearn/TF-IDF: normalize whitespace and lowercase for stability.
"""

import re
from typing import Iterable, List

_ws_re = re.compile(r"\s+")


def _basic_clean(text: str) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r", " ").replace("\n", " ").strip()
    text = _ws_re.sub(" ", text)
    return text


def for_transformer(text: str) -> str:
    return _basic_clean(text)


def for_sklearn(text: str) -> str:
    return _basic_clean(text).lower()


def batch_prep(texts: Iterable[str], kind: str) -> List[str]:
    if kind == "transformer":
        return [for_transformer(t) for t in texts]
    return [for_sklearn(t) for t in texts]
