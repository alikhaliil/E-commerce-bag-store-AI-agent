# Haqiba Bags AI Support Agent - production image
#   docker build -t haqiba-agent .
#   docker run --rm -p 8000:8000 --env-file .env haqiba-agent
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    ANONYMIZED_TELEMETRY=False

WORKDIR /app

# CPU-only PyTorch first. The default wheel bundles CUDA libraries (~2.5 GB)
# that a CPU embedding model never uses; this keeps the image roughly 1.5 GB smaller.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY frontend ./frontend
COPY data/store_policies.md ./data/store_policies.md

# Bake the embedding model, the seeded SQLite DB and the Chroma index into the
# image, so a cold start on Render / HF Spaces serves the first request quickly.
# The app still self-heals at startup if any of them is missing.
RUN python -m app.db.seed && python -m app.rag.ingest

# Run as a non-root user (uid 1000 also matches Hugging Face Spaces).
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=4)" || exit 1

# ${PORT} lets platforms like Render inject their port; defaults to 8000.
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
