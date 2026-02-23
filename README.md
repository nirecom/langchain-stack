# LangChain Stack

LLM-as-a-Judge chain with RAG support, exposed as an OpenAI-compatible API.

## Architecture

```
Open WebUI ───┐
Other tools ──┤──→ langchain-api:8100 ──→ LiteLLM Proxy ──→ LLM backends
              │         │
              │         └──→ ChromaDB:8200 (RAG vector search)
              │
              └──→ LiteLLM Proxy:4000 (direct model access)
```

All LLM calls go through an external LiteLLM Proxy on LAN. No direct model endpoints in this stack.

## Quick Start

```bash
git clone <this-repo> && cd langchain-stack
cp .env.example .env
vi .env  # Fill in LiteLLM proxy URL and API key
docker compose up -d
```

## Verify

```bash
# Health check
curl http://localhost:8100/health

# Test reasoner passthrough
curl http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "judge-chain", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `LITELLM_PROXY_URL` | LiteLLM Proxy endpoint | `http://<your-litellm-host>:4000/v1` |
| `LITELLM_API_KEY` | LiteLLM master key | `sk-xxxxx` |
| `LANGCHAIN_API_PORT` | API server port (default: 8100) | `8100` |
| `CHROMA_PORT` | ChromaDB exposed port (default: 8200) | `8200` |
| `MAX_JUDGE_RETRIES` | Max retry attempts for judge chain (default: 2) | `2` |

## Implementation Phases

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 2 | ✅ Current | Skeleton — reasoner passthrough via LiteLLM |
| Phase 3 | TODO | Judge evaluation with retry loop |
| Phase 4 | TODO | RAG (ChromaDB ingestion + retrieval) |

## Daily Operations

```bash
docker compose ps                             # Check status
docker compose logs -f                        # Follow logs
docker compose build && docker compose up -d  # Rebuild after code change
docker compose down                           # Stop (data preserved)
```

## ⚠ Caution

- `.env` contains secrets — it is excluded from Git via `.gitignore`
- `data/` directory is gitignored (runtime data: ChromaDB, documents)
- Never use `docker compose down -v` (deletes ChromaDB data)
