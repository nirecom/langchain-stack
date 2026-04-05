FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# System packages for document loaders
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 poppler-utils antiword \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-editable

COPY . .

EXPOSE 8100

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8100", "--loop", "asyncio"]
