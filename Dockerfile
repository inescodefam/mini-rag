FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    # Safe default: fake embeddings + stub LLM, no AWS and no cost.
    # Run with -e MINIRAG_OFFLINE=0 to use Bedrock.
    MINIRAG_OFFLINE=1

WORKDIR /app

# Install dependencies first so this layer is cached when only code changes.
# pytest and httpx are test-only, but they are pinned in the same file.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY corpus ./corpus

# Do not run as root. No AWS credentials are baked into the image.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "minirag.api:app", "--host", "0.0.0.0", "--port", "8000"]
