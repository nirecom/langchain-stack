# LangChain Stack

LLM-as-a-Judge chain with RAG support, exposed as an OpenAI-compatible API.

## Features

- **LLM-as-a-Judge**: Evaluates LLM responses with verdict, score, and feedback. Supports retry loops.
- **RAG**: ChromaDB-backed retrieval with per-user datasource access control. Uses `BAAI/bge-m3` embedding (selected via A/B evaluation against ruri/qwen3/bgem3).
- **User-based ACL**: Each user has an individual API key mapped to permitted datasources.
- **Streaming**: SSE streaming (`stream: true`) supported.
- **Audit logging**: Ingest and retrieval events logged to JSONL with user attribution.
- **OpenAI-compatible**: Drop-in replacement endpoint for chat completions.

## Architecture

```
Open WebUI ───┐
Other tools ──┤──→ langchain-api:8100 ──→ LiteLLM Proxy ──→ LLM backends
              │         │
              │         └──→ ChromaDB:8200 (RAG vector search)
```

All LLM calls go through an external LiteLLM Proxy on LAN.

## Quick Start

```bash
git clone <this-repo> && cd langchain-stack
cp .env.example .env
# Fill in API keys (see .env.example for all variables)
docker compose up -d
```

## Verify

```bash
curl http://localhost:8100/health
```

## Environment Variables

See `.env.example` for the full list. Key variables:

| Variable | Description |
|----------|-------------|
| `LITELLM_PROXY_URL` | LiteLLM Proxy endpoint |
| `LITELLM_API_KEY` | LiteLLM master key |
| `INGEST_API_KEY` | Admin key for document ingestion |
| `CHAT_API_KEY_<USER>` | Per-user chat API key (KYOKO, NIRE, EDGE, LUTE) |
| `EMBEDDING_MODEL_NAME` | Embedding model (default: `BAAI/bge-m3`) |
| `INGEST_DEVICE` | Device for ingest embeddings: `cpu` (default) or `cuda` |

## Daily Operations

```bash
docker compose ps
docker compose logs -f langchain-api
docker compose up -d langchain-api   # Restart after config/code change
```

## ⚠ Caution

- `.env` contains secrets — excluded from Git via `.gitignore`. Copy from `.env.example` and fill in values.
- Changing `EMBEDDING_MODEL_NAME` requires re-ingesting all documents (vectors are model-specific)
- `data/` directory is gitignored (ChromaDB, uploads, audit logs)
- Never use `docker compose down -v` (deletes ChromaDB data)
