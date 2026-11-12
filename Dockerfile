
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY models ./models
COPY run_server.py .
COPY README.md .

EXPOSE 7860

CMD ["sh", "-c", "uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-7860}"]