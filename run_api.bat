@echo off
call conda activate twitter-sentiment-env
python -m uvicorn src_2.app:app --host 0.0.0.0 --port 8000 --reload
