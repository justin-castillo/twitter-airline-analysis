"""
Centralized, robust path handling for the project.
Works even when run via Streamlit or Uvicorn.
"""
from pathlib import Path

# This file lives in <project_root>/src_2/paths.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Common top-level folders in your project
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" 

DISTILBERT_DIR = MODELS_DIR / "distilbert_twitter" / "final"

# Candidate sklearn pipelines (prefer the optuna one if present)
SKLEARN_CANDIDATES = [
    MODELS_DIR / "logreg_tfidf_optuna.joblib",
    MODELS_DIR / "logreg_tfidf.joblib",
]

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

DEFAULT_SKLEARN_PIPELINE = first_existing(SKLEARN_CANDIDATES)
