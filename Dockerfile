FROM python:3.12.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ZANIA_MODEL_CACHE=/app/.models \
    HF_HUB_OFFLINE=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

# Public, revision-pinned model download at build time; no API credentials needed.
COPY scripts/prepare_model.py scripts/prepare_model.py
RUN HF_HUB_OFFLINE=0 python scripts/prepare_model.py --directory /app/.models
COPY app/ app/

RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
