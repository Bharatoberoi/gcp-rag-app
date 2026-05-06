FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Pre-download the embedding model during build so startup is instant
ENV SENTENCE_TRANSFORMERS_HOME=/app/models
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

RUN mkdir -p /data && \
    groupadd --system rag && useradd --system --gid rag rag \
    && chown -R rag:rag /app /data

USER rag

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=8s --start-period=60s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8080}/health" >/dev/null || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
