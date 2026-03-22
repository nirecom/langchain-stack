# CLAUDE.md

## Project Overview

LLM-as-a-Judge orchestration system with RAG support, exposed as an OpenAI-compatible API.

## Design Documents (via additionalDirectories)

Read these files at the start of every implementation session:
- `../ai-specs/projects/engineering/langchain/architecture.md`
- `../ai-specs/projects/engineering/langchain/progress.md`

Do NOT read files outside of `../ai-specs/projects/engineering/`.
ai-specs is a PRIVATE repository — never reference its content
in commits to this public repository.

## Key Architecture Decisions

- **Settings**: `app/settings.py` — Pydantic Settings loads env vars + YAML configs (`config/models.yaml`, `config/judge_rules.yaml`) at startup
- **LLM Provider Isolation**: All model instantiation in `app/models/provider.py` via `get_reasoner()`/`get_judge()` factory functions using LangChain's `ChatOpenAI`. Direct endpoint fallback probes llama-swap before each chain, falling back to LiteLLM when all are down
- **Orchestration**: `app/chains/llm_as_judge.py` — Reasoner → RAGAS Response Relevancy evaluation pipeline
- **Judge Feedback**: `app/chains/judge.py` — `generate_feedback()` for FAIL retry
- **RAGAS Evaluation**: `app/evaluation/metrics.py` — RAGAS Response Relevancy scorer (Judge LLM for question generation + ruri embeddings for similarity)
- **RAG Retriever**: `app/rag/retriever.py` — ChromaDB stub (Phase 4)
- **API**: `app/main.py` — FastAPI with `/health` and `/v1/chat/completions` (OpenAI-compatible format with added `metadata` field for judge verdict/score)
- **Evaluation Visibility**: `app/main.py` `format_judge_evaluation()` — Appends collapsible `<details>` section with per-attempt scores/verdicts to response content
- **SSE Streaming**: `app/chains/llm_as_judge.py` `run_judge_chain_stream()` — Async generator yielding status/token/evaluation events; `app/main.py` `_stream_response()` formats as OpenAI `chat.completion.chunk` SSE

## Coding Conventions

- No hardcoded IPs, ports, or model names — use env vars or YAML config
- Async/await throughout (FastAPI + LangChain `ainvoke`)
- `.env` and `data/` are gitignored; no secrets in committed files
- Configuration: `.env` for secrets/endpoints, `config/*.yaml` for model and evaluation settings
