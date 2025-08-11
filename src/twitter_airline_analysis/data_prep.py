"""
Data‑preparation utilities for Twitter Airline Sentiment project.
All functions are pure and side‑effect free except `save_*`.
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = RAW_DIR.parent / "processed"
PROCESSED_DIR.mkdir(exist_ok=True)

_URL_RE = re.compile(r"https?://\S+")
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#\w+")


def load_raw(csv_name: str = "Tweets.csv") -> pd.DataFrame:
    """Read raw CSV into a DataFrame."""
    return pd.read_csv(RAW_DIR / csv_name)


def clean_text(text: str) -> str:
    """Lower‑case, normalise unicode, and strip URLs, mentions, hashtags."""
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = _URL_RE.sub("", text)
    text = _MENTION.sub("", text)
    text = _HASHTAG.sub("", text)
    return text.strip()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning and return a tidy DataFrame."""
    df = df.copy()
    df["clean_text"] = df["text"].astype(str).apply(clean_text)
    cols = ["tweet_id", "airline", "airline_sentiment", "clean_text", "negativereason"]
    return df[cols]


def save_parquet(df: pd.DataFrame, name: str = "tweets.parquet") -> Path:
    """Save DataFrame as Parquet and return path."""
    out_path = PROCESSED_DIR / name
    df.to_parquet(out_path, index=False)
    return out_path


def load_prepared_data(*, force_rebuild: bool = False):
    """
    Return (X, y) ready for modelling.

    Parameters
    ----------
    force_rebuild : bool, default=False
        If True, always rebuild the processed parquet from raw CSV.

    Notes
    -----
    • Looks for `tweets.parquet` in the processed folder.
    • If missing or `force_rebuild=True`, runs the full preprocess pipeline.
    """
    parquet_path = PROCESSED_DIR / "tweets.parquet"

    if force_rebuild or not parquet_path.exists():
        df_raw = load_raw()
        df_tidy = preprocess(df_raw)
        parquet_path = save_parquet(df_tidy)

    df = pd.read_parquet(parquet_path)

    X = df["clean_text"]
    y = df["airline_sentiment"]

    return X, y


if __name__ == "__main__":  # manual CLI
    df_raw = load_raw()
    df_tidy = preprocess(df_raw)
    path = save_parquet(df_tidy)
    print(f"Saved {len(df_tidy):,} rows → {path}")
