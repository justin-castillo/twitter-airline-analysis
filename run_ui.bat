@echo off
call conda activate twitter-sentiment-env
python -m streamlit run "%~dp0src_2\streamlit_app.py" --server.port 8501
