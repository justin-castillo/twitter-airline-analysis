# syntax=docker/dockerfile:1

# ---------- build stage ----------
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY . .

# ---------- runtime stage ----------
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

# copy only installed site‑packages + source
COPY --from=builder /root/.local /usr/local
COPY --from=builder /app ./

CMD ["uvicorn", "inference_nodes.service:app", "--host", "0.0.0.0", "--port", "8000"]
