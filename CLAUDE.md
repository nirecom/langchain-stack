# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM-as-a-Judge orchestration system with RAG support, exposed as an OpenAI-compatible API. Currently in **Phase 3D** (Judge evaluation visibility in open-webui).

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
- **Orchestration**: `app/chains/llm_as_judge.py` — Reasoner → RAGAS Response Relevancy evaluation pipeline
- **Judge Feedback**: `app/chains/judge.py` — `generate_feedback()` for FAIL retry (Phase 3C)
- **RAGAS Evaluation**: `app/evaluation/metrics.py` — RAGAS Response Relevancy scorer (Judge LLM for question generation + ruri embeddings for similarity)
- **RAG Retriever**: `app/rag/retriever.py` — ChromaDB stub for Phase 4
- **API**: `app/main.py` — FastAPI with `/health` and `/v1/chat/completions` (OpenAI-compatible format with added `metadata` field for judge verdict/score)
- **Evaluation Visibility**: `app/main.py` `format_judge_evaluation()` — Appends collapsible `<details>` section with per-attempt scores/verdicts to response content (Phase 3D)

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
| 3E | Next | SSE streaming for real-time evaluation progress |
| 4 | Future | RAG: ChromaDB ingestion + vector search retrieval |

## Phase Lifecycle

> Canonical rules: `../ai-specs/CLAUDE.md` § "LangChain Project: Phase Workflow"

### Starting a Phase

1. Read the handoff document `.context-private/handoff-phase{N}.md`
2. Read the previous phase completion report `.context-private/completion-phase{prev}.md`
3. Read PJ global docs (listed in "Overall Architecture Documents" above) + this `CLAUDE.md`
4. Cross-check handoff against higher-priority documents for contradictions:
   - **Priority 1 (PJ global docs)**: `langchain-design.md`, `langchain-setup-progress.md`, `CLAUDE.md` — these may have been updated after the handoff was written. On conflict, PJ global docs win.
   - **Priority 2**: Previous phase completion report (`completion-phase{prev}.md`)
   - **Priority 3**: Current handoff (`handoff-phase{N}.md`)
5. If a conflict cannot be resolved by priority alone, ask the user — do not decide independently
6. Present corrections and plan before writing code

### Completing a Phase

1. Verify all completion criteria from the handoff document
2. Update the Implementation Phases table in this file
3. Create `.context-private/completion-phase{N}.md` containing:
   - Completed work (files changed per commit)
   - Verification results
   - Corrections applied to handoff
   - Current system state (API format, config, etc.)
   - Notes and warnings for the next phase
4. Update PJ global docs: checklist in `langchain-design.md`, progress in `langchain-setup-progress.md`
5. Commit to develop branch and push
