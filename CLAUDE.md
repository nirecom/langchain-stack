# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM-as-a-Judge orchestration system with RAG support, exposed as an OpenAI-compatible API. Currently in **Phase 2** (reasoner passthrough via LiteLLM Proxy; no judge evaluation yet).

## Overall Architecture Documents (via additionalDirectories)

Read these files at the start of every implementation session:
- ../ai-specs/projects/engineering/langchain-design.md
- ../ai-specs/projects/engineering/langchain-setup-progress.md

Do NOT read files outside of ../ai-specs/projects/engineering/.
ai-specs is a PRIVATE repository — never reference its content
in commits to this public repository.

## LangChain Architecture

```
Clients ──→ langchain-api:8100 (FastAPI) ──→ LiteLLM Proxy (LAN) ──→ llama-swap models
                     │
                     └──→ ChromaDB:8200 (RAG vector search, Phase 4)
```

All LLM calls route through an external LiteLLM Proxy — no direct model endpoints in this stack. Inside Docker, use the LAN IP for LiteLLM (not localhost).

## Build and Run

```bash
cp .env.example .env          # then fill in LiteLLM proxy URL and API key
docker compose up -d --build  # build and start services
docker compose logs -f        # follow logs
docker compose build && docker compose up -d  # rebuild after code changes
```

**Never use `docker compose down -v`** — it deletes ChromaDB persistent data.

## Verify

```bash
curl http://localhost:8100/health
curl http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"judge-chain","messages":[{"role":"user","content":"Hello"}]}'
```

## Key Architecture Decisions

- **Settings**: `app/settings.py` — Pydantic Settings loads env vars + YAML configs (`config/models.yaml`, `config/judge_rules.yaml`) at startup
- **LLM Provider Isolation**: All model instantiation in `app/models/provider.py` via `get_reasoner()`/`get_judge()` factory functions using LangChain's `ChatOpenAI`
- **Orchestration**: `app/chains/llm_as_judge.py` — Phase 2 passes through to reasoner; Phase 3 will add judge evaluation with retry loop
- **RAG Retriever**: `app/rag/retriever.py` — ChromaDB stub for Phase 4
- **API**: `app/main.py` — FastAPI with `/health` and `/v1/chat/completions` (OpenAI-compatible format with added `metadata` field for judge verdict/score)

## Configuration

- **Environment** (`.env`): `LITELLM_PROXY_URL`, `LITELLM_API_KEY`, `LANGCHAIN_API_PORT` (8100), `CHROMA_PORT` (8200), `MAX_JUDGE_RETRIES` (2)
- **`config/models.yaml`**: Logical model definitions (reasoner at temp 0.7, judge at temp 0.0) mapped to LiteLLM model names
- **`config/judge_rules.yaml`**: Evaluation criteria rulesets — `default` (general quality) and `rag` (faithfulness/grounding)

## Coding Conventions

- No hardcoded IPs, ports, or model names — use env vars or YAML config
- Async/await throughout (FastAPI + LangChain `ainvoke`)
- English for all code, comments, and commit messages
- `.env` and `data/` are gitignored; no secrets in committed files

## Implementation Phases

| Phase | Status | Scope |
|-------|--------|-------|
| 2 | Current | Reasoner passthrough, no judge evaluation |
| 3 | Next | Judge evaluation + retry loop |
| 4 | Future | RAG: ChromaDB ingestion + vector search retrieval |
