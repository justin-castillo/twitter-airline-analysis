FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR /app

COPY requirements.deploy.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src_2/ /app/src_2/
COPY models/ /app/models/
ENV PYTHONPATH=/app
EXPOSE 8501
CMD ["sh","-c","streamlit run src_2/streamlit_app.py --server.address=0.0.0.0 --server.port ${PORT:-8501}"]
