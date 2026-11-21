FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install backend dependencies directly for a self-contained container build
RUN pip install --no-cache-dir \
    fastapi==0.115.6 \
    uvicorn[standard]==0.34.0 \
    sqlalchemy==2.0.36 \
    pydantic==2.10.4 \
    numpy==2.2.1 \
    google-genai==1.3.0

# Copy application source and dashboard
COPY models.py engine.py main.py index.html ./

# Ensure persistent SQLite mount directory exists
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]