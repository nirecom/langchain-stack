# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM-as-a-Judge orchestration system with RAG support, exposed as an OpenAI-compatible API. Currently in **Phase 3E** (SSE streaming for real-time evaluation progress).

## Overall Architecture Documents (via additionalDirectories)

Read these files at the start of every implementation session:
- ../ai-specs/projects/engineering/langchain/architecture.md
- ../ai-specs/projects/engineering/langchain/progress.md

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
- **Orchestration**: `app/chains/llm_as_judge.py` — Reasoner → RAGAS Response Relevancy evaluation pipeline
- **Judge Feedback**: `app/chains/judge.py` — `generate_feedback()` for FAIL retry (Phase 3C)
- **RAGAS Evaluation**: `app/evaluation/metrics.py` — RAGAS Response Relevancy scorer (Judge LLM for question generation + ruri embeddings for similarity)
- **RAG Retriever**: `app/rag/retriever.py` — ChromaDB stub for Phase 4
- **API**: `app/main.py` — FastAPI with `/health` and `/v1/chat/completions` (OpenAI-compatible format with added `metadata` field for judge verdict/score)
- **Evaluation Visibility**: `app/main.py` `format_judge_evaluation()` — Appends collapsible `<details>` section with per-attempt scores/verdicts to response content (Phase 3D)
- **SSE Streaming**: `app/chains/llm_as_judge.py` `run_judge_chain_stream()` — Async generator yielding status/token/evaluation events; `app/main.py` `_stream_response()` formats as OpenAI chat.completion.chunk SSE (Phase 3E)

## Configuration

- **Environment** (`.env`): `LITELLM_PROXY_URL`, `LITELLM_API_KEY`, `LANGCHAIN_API_PORT` (8100), `CHROMA_PORT` (8200), `MAX_JUDGE_RETRIES` (2), `RAGAS_RESPONSE_RELEVANCY_THRESHOLD` (0.7)
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
| 2 | Done | Reasoner passthrough via LiteLLM |
| 3A | Done | Reasoner → Judge LLM evaluation (no retry) |
| 3B | Done | RAGAS Response Relevancy quantitative scoring |
| 3C | Done | Judge feedback + retry loop |
| 3D | Done | Judge evaluation visibility in response content |
| 3E | Done | SSE streaming for real-time evaluation progress |
| 4 | Future | RAG: ChromaDB ingestion + vector search retrieval |

## Phase Lifecycle

> Canonical rules: `../ai-specs/CLAUDE.md` § "LangChain Project: Phase Workflow"

Use `/start-langchain-task {ID}` to begin and `/complete-langchain-task {ID}` to finish.
Phase handoffs and completion reports are stored in `.context-private/`.

### Conflict resolution

- **Priority 1**: PJ global docs (`architecture.md`, `progress.md`, `CLAUDE.md`) — on conflict, these win
- **Priority 2**: Previous phase completion report
- **Priority 3**: Current handoff document
- If unresolvable by priority, ask the user
