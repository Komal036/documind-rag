FROM python:3.11-slim

WORKDIR /app

# System deps some of your packages need (psycopg2, torch, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run injects $PORT — must listen on it
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port $PORT"]