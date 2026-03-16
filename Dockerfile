FROM python:3.11-slim-bullseye

WORKDIR /app

# Install system dependencies needed for FAISS and asyncpg
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:0.5.0 /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml .
COPY opsagent/ ./opsagent/

# Install dependencies (no dev extras in prod)
RUN uv pip install --system --no-cache -e .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "opsagent.main:app", "--host", "0.0.0.0", "--port", "8000"]
