# Twitter US Airline Sentiment — End‑to‑End NLP Pipeline

This repository delivers an end‑to‑end sentiment‑analysis workflow for the **Twitter US Airline Sentiment** dataset (≈ 14,600 tweets, Feb 2015): data prep → EDA → **TF‑IDF + Logistic Regression** → **DistilBERT** fine‑tune → evaluation & interpretability → **Streamlit demo** and **FastAPI** service → **Docker + CI** for production‑style packaging.

---

## Key results (held‑out test)

| Model | Accuracy | Macro‑F1 | Notes |
|---|---:|---:|---|
| **DistilBERT (fine‑tuned)** | **0.836** | **0.788** | Strong ROC/PR; main confusions are *negative ↔ neutral*. |
| **TF‑IDF + Logistic Regression (tuned)** | 0.790 | 0.739 | Macro ROC‑AUC ≈ 0.907; transparent token weights; good fallback. |

**Operational guidance.** Errors concentrate at mid confidences (~0.40–0.70); consider **auto‑route ≥ 0.80** and **human‑review 0.50–0.79** in production.

---

## Dataset

- **Source.** Kaggle — *Twitter US Airline Sentiment* (Feb–Mar 2015).
- **Labels.** `negative`, `neutral`, `positive`, plus an optional `negativereason` field for error analysis.
- **Imbalance.** ~63 % negative, 21 % neutral, 16 % positive. Use stratified splits and class weighting for classical models.
- **File layout.** Place the raw CSV at `data/raw/Tweets.csv`. Processed parquet and feather splits are written to `data/processed/`.

---

## Repository layout

```
notebooks/                    # EDA → modeling → deployment (PDF exports in repo)
models/
  distilbert_twitter/final/   # Fine‑tuned DistilBERT checkpoint + tokenizer
  logreg_tfidf.joblib         # Baseline sklearn pipeline
  logreg_tfidf_optuna.joblib  # (optional) tuned sklearn pipeline
src_2/
  app.py                      # FastAPI service (POST /predict)
  streamlit_app.py            # Streamlit demo UI
  model.py, predict.py        # Unified loader + thin inference API
  data_prep.py, paths.py      # Lightweight text prep + robust project paths
reports/                      # Metrics JSON + evaluation figures
Dockerfile                    # Multi‑stage image (Python 3.11‑slim)
.github/workflows/ci.yml      # Lint → tests → docker build → optional push on tags
```

---

## Pipeline overview

### 1) Data preparation & EDA
- Minimal, robust cleaning (normalize whitespace; lowercase only for sklearn paths).  
- Cleaned data are saved to **Parquet**; stratified **Feather splits** `X_*` / `y_*` are produced for `train`/`val`/`test` (11,712 / 1,464 / 1,464).  
- EDA: class balance, lengths, negative‑reason categories, and airline‑level sentiment.

### 2) Classical baseline — TF‑IDF + Logistic Regression
- Bi‑gram TF‑IDF (`max_features` up to 50k) + `LogisticRegression(class_weight='balanced')`.  
- Validation ≈ 0.79 accuracy; macro‑F1 ≈ 0.74.  
- **Optuna tuning** jointly searches TF‑IDF and LR hyper‑parameters; the best pipeline is persisted as `models/logreg_tfidf_optuna.joblib`.  
- Diagnostics: confusion matrix, ROC, token‑importance bars, confidence histograms, t‑SNE, lift curves.

### 3) Transformer — DistilBERT
- Base: `distilbert-base-uncased`, max length 128, batch 16, LR 2e‑5, **2 epochs**.  
- Test results ≈ 0.84 accuracy / ≈ 0.79 macro‑F1; negative/positive are cleanly separated, neutral is harder.  
- Final operating policy derived from calibration & confidence diagnostics.

---

## Inference

### A) Streamlit (local demo)
```bash
streamlit run src_2/streamlit_app.py
```
Use the sidebar to switch between **distilbert** and **sklearn**, and to return all class scores.

### B) FastAPI (programmatic)
```bash
uvicorn src_2.app:app --host 0.0.0.0 --port 8000
# POST /predict with either payload
# { "text": "The flight was delayed" }
# { "texts": ["Great staff!", "Late again :("], "model_name": "distilbert", "return_all_scores": true }
```

### C) From Python
```python
from src_2.predict import predict_texts
predict_texts("Great service!", model_name="distilbert", return_all_scores=True)
```

---

## Docker & CI

### Build locally
```bash
docker build -t airline-sentiment:latest .
docker run -p 8000:8000 airline-sentiment:latest
# Then POST to http://127.0.0.1:8000/predict
```

The Dockerfile uses a multi‑stage **Python 3.11‑slim** base and launches Uvicorn with `src_2.app:app` as the entrypoint.

### GitHub Actions
The CI workflow lints & tests the repo, builds the Docker image on every commit, and (on tags like `v*.*.*`) can push to a container registry.

---

## Reproducibility & artifacts

- **Models:** `models/distilbert_twitter/final/`, `models/logreg_tfidf*.joblib`  
- **Metrics & figures:** confusion matrices, ROC/PR, calibration, t‑SNE, confidence histograms, lift curves under `reports/`

---

## Next steps

- Apply temperature scaling for tighter probability calibration.  
- Collect / augment **neutral** examples; consider class‑weighted or focal loss.  
- Explore domain‑specific LMs (e.g., TweetEval‑style) if neutral recall remains the main pain point.

---

*This README reflects the implemented stack (TF‑IDF + LogReg and DistilBERT), the shipped demo apps (Streamlit + FastAPI), and the packaging (Docker + CI).*

