### Inference Usage

```bash
# CLI
python -m src.predict "Flight delayed again :("   # → negative
```

```bash
# API (local dev)
uvicorn src.app:app --host 0.0.0.0 --port 8000
# then:
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{ "text": "Great service!" }'
# → {"label":"positive"}
```
