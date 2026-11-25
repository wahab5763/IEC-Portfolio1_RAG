# ==============================================================================
# Multi-Stage Production Dockerfile for Render Free Tier (FastAPI + React SPA)
# ==============================================================================

# --- Stage 1: Build React + Vite + Tailwind SPA ---
FROM node:22-alpine AS frontend-builder
WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


# --- Stage 2: Lean Python 3.12 Runtime ---
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken_cache

WORKDIR /app

# Install libmagic for Linux content-signature validation and clean apt cache
RUN apt-get update \
    && apt-get install -y --no-install-recommends libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Install lean production Python dependencies
COPY requirements-prod.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements-prod.txt

# Pre-warm tiktoken cl100k_base BPE vocabulary during image build (zero cold-start lag)
RUN mkdir -p /app/.tiktoken_cache \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# Copy backend code, sample corpus (for Load Sample Corpus button), and built React SPA
COPY backend ./backend
COPY data/corpus ./data/corpus
COPY --from=frontend-builder /frontend/dist ./frontend/dist

# Run as non-root user for container security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Render injects $PORT automatically; defaults to 8000 for local Docker runs
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
