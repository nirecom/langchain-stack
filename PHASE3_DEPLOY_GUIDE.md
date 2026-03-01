# Phase 3 Implementation Guide: LLM-as-a-Judge Chain

## Overview

Phase 3 transforms the Phase 2 passthrough skeleton into a hybrid LLM-as-a-Judge
chain using **RAGAS Response Relevancy** for quantitative scoring and **Judge LLM**
for qualitative feedback generation.

## Architecture: Hybrid Scoring Flow

```
User Question
    │
    ▼
┌──────────────┐
│ Reasoner LLM │ → penpen 120B (via LiteLLM)
│ (generate)   │
└──────┬───────┘
       │ answer
       ▼
┌─────────────────────────────┐
│ RAGAS Response Relevancy    │
│  LLM: judge model (7B)     │   ← Reverse-generates N questions
│  Embeddings: e5-small (CPU) │   ← Cosine similarity vs original
│  Score: 0.0 ~ 1.0          │
└──────┬──────────────────────┘
       │
   score >= 0.7? ──YES──→ PASS → Return answer
       │
      NO (FAIL)
       │
       ▼
┌──────────────┐
│ Judge LLM    │ → Mac/penpen 7B (via LiteLLM)
│ (feedback)   │   "Answer is too vague, needs..."
└──────┬───────┘
       │ feedback
       ▼
   Reasoner LLM (retry with feedback)
       │
       ▼
   RAGAS (re-score) → PASS? → Return best answer
       │
   (max 2 retries)
```

Key design decisions:
- **PASS/FAIL** is determined solely by RAGAS quantitative score (reproducible)
- **Judge LLM** is only invoked on FAIL to generate actionable feedback (cost-efficient)
- **Best answer** across all attempts is returned (even if all attempts FAIL)

## File Changes from Phase 2 Skeleton

### New files

| File | Purpose |
|------|---------|
| `app/evaluation/__init__.py` | Evaluation module package |
| `app/evaluation/metrics.py` | RAGAS Response Relevancy scorer integration |
| `app/chains/reasoner.py` | Reasoner chain (answer generation + retry) |
| `app/chains/judge.py` | Judge feedback chain (FAIL-only feedback generation) |

### Modified files

| File | Changes |
|------|---------|
| `config/judge_rules.yaml` | Rewritten: RAGAS-aligned profiles (default/rag) with metric config |
| `app/requirements.txt` | Added: `ragas>=0.2,<0.3`, `sentence-transformers>=3.0` |
| `app/settings.py` | Added: RAGAS embedding model config, judge profile accessors |
| `app/models/provider.py` | Added: `temperature` None-check for explicit 0.0 support |
| `app/chains/llm_as_judge.py` | Rewritten: Hybrid RAGAS score + Judge LLM feedback loop |
| `app/main.py` | Updated: RAGAS metadata in response, profile auto-selection |
| `app/Dockerfile` | Added: build-essential, sentence-transformers model pre-download |

### Unchanged files

| File | Notes |
|------|-------|
| `config/models.yaml` | No changes (reasoner/judge logical names stay the same) |
| `docker-compose.yml` | No changes needed |
| `.env` / `.env.example` | No new env vars required for Phase 3 |
| `app/rag/retriever.py` | Stub returns empty string (Phase 4) |

## Deployment Steps

### Step 1: Copy files to penpen

Copy the Phase 3 files to the langchain-stack repo on penpen, replacing
the Phase 2 skeleton versions:

```powershell
cd C:\LLM\langchain-stack

# Backup current state
git stash

# Copy new/modified files (from wherever you've placed the Phase 3 output)
# Key files to copy:
#   config/judge_rules.yaml
#   app/requirements.txt
#   app/settings.py
#   app/Dockerfile
#   app/main.py
#   app/models/provider.py
#   app/chains/reasoner.py      (NEW)
#   app/chains/judge.py         (NEW)
#   app/chains/llm_as_judge.py
#   app/evaluation/__init__.py  (NEW)
#   app/evaluation/metrics.py   (NEW)
```

### Step 2: Rebuild Docker image

The image rebuild will take longer than usual due to `sentence-transformers`
and the embedding model pre-download (~5-10 minutes on first build).

```powershell
cd C:\LLM\langchain-stack
docker compose up -d --build
```

Watch the build log for the model download step:

```powershell
docker compose logs -f langchain-api
```

### Step 3: Verify health

```powershell
curl.exe http://localhost:8100/health
# Expected: {"status":"ok","version":"0.3.0"}
```

### Step 4: Test the Judge chain

```powershell
# Test with a clear, answerable question (should PASS)
curl.exe http://localhost:8100/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{\"model\":\"judge-chain\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France?\"}]}'
```

Expected response includes RAGAS metadata:

```json
{
  "choices": [{"message": {"content": "..."}}],
  "metadata": {
    "judge_verdict": "PASS",
    "response_relevancy": 0.85,
    "retries": 0,
    "judge_feedback": ""
  }
}
```

### Step 5: Test FAIL → retry flow

Use a vague or trick question to trigger low Response Relevancy:

```powershell
curl.exe http://localhost:8100/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{\"model\":\"judge-chain\",\"messages\":[{\"role\":\"user\",\"content\":\"Tell me everything.\"}],\"temperature\":1.0}'
```

Check logs for the retry flow:

```powershell
docker compose logs langchain-api | Select-String "Response Relevancy"
```

## Configuration Reference

### judge_rules.yaml thresholds

| Parameter | Default | Description |
|-----------|---------|-------------|
| `response_relevancy.pass_threshold` | 0.7 | RAGAS score below this = FAIL |
| `response_relevancy.strictness` | 3 | Number of reverse-generated questions |
| `max_judge_retries` (env) | 2 | Max retry attempts on FAIL |

### Tuning guidance

- **threshold too high (>0.85)**: Most answers will FAIL, causing excessive retries
  and LLM cost. Start at 0.7 and increase based on observed score distribution.
- **threshold too low (<0.5)**: Nearly everything PASSes, Judge adds no value.
- **strictness**: 3 is the RAGAS default. Increasing to 5 improves score stability
  but adds 2 extra LLM calls per evaluation.

## Troubleshooting

### "Failed to compute Response Relevancy" in logs

The RAGAS scorer failed. Common causes:
1. LiteLLM proxy is unreachable from the langchain-api container
2. Judge model (Qwen2.5-7B) returned malformed output
3. Embedding model failed to load

Check: `docker exec langchain-api python -c "from evaluation.metrics import _get_ragas_embeddings; print(_get_ragas_embeddings())"`

### Score is always 0.0

The fallback on error returns 0.0. Check langchain-api logs for exceptions.
Most likely the LiteLLM connection is failing.

### Docker build fails at sentence-transformers

The `sentence-transformers` package requires `torch`. If build fails on ARM
or memory-constrained hosts, try adding `--platform linux/amd64` to the build.

### First request is very slow

The embedding model loads into memory on first use (~5 seconds). Subsequent
requests use the cached singleton. The Dockerfile pre-downloads the model
weights, but the in-memory loading still happens at first request.
