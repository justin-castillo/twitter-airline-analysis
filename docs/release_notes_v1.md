# v1.0.0 — 2025-08-04

**Highlights**

* End‑to‑end packaging: Dockerfile + FastAPI micro‑service.
* Automated CI/CD via GitHub Actions (lint → tests → image build → GHCR push).
* Model performance on held‑out test set  
  | Metric | Value |  
  |---|---|  
  | Accuracy | **0.790** |  
  | Macro F1 | **0.739** |  
  | Macro ROC AUC | **0.907** |

**Install & Run**

```bash
docker pull ghcr.io/<your‑org>/airline-sentiment:v1.0.0
docker run -p 8000:8000 airline-sentiment:v1.0.0
# → POST text to http://localhost:8000/predict
```
