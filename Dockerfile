FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/.cache \
    && chown -R appuser:appuser /app

USER appuser

ENV ST_HOST=0.0.0.0
ENV ST_PORT=8000
ENV ST_MODEL_CACHE_DIR=/app/.cache
ENV ST_MODEL_NAME=BAAI/bge-m3
ENV ST_DEVICE=cpu

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "st_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
