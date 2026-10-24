# Lightweight Streamlit Frontend Dockerfile
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY frontend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ ./frontend/
COPY backend/ ./backend/
COPY models/ ./models/
COPY data/processed/ ./data/processed/

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
