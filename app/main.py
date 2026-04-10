"""
LangChain API Server — OpenAI-compatible endpoint.
Exposes LLM-as-a-Judge chain via /v1/chat/completions.
Phase 3E: SSE streaming support.
Phase 4A: Ingestion endpoints.
Phase 4B: Access control, audit logging, dry-run mode.
"""
import json
import logging
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from chains.llm_as_judge import run_judge_chain, run_judge_chain_stream
from rag.retriever import get_relevant_context
from rag.access_control import is_valid_datasource
from rag.audit import log_ingest_event, get_recent_events
import chromadb.errors
from rag.ingest import (
    ingest_file, ingest_folder, delete_collection,
    list_files, delete_file, dry_run_file, SUPPORTED_EXTENSIONS,
)
from settings import settings

logger = logging.getLogger(__name__)


# --- Phase 4B: Authentication & authorization helpers ---


def _verify_api_key(request: Request, expected_key: str) -> None:
    """Check Bearer token. Raises 401 if key is set and token is missing/wrong."""
    if not expected_key:
        return  # Auth disabled
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != expected_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _verify_ingest_auth(request: Request) -> None:
    _verify_api_key(request, settings.ingest_api_key)


def _verify_chat_auth(request: Request) -> None:
    _verify_api_key(request, settings.chat_api_key)


def _validate_datasource(datasource: str) -> None:
    """Reject datasource not registered in access_control.yaml."""
    if not is_valid_datasource(datasource):
        log_ingest_event("ingest", datasource, status="error", error="unregistered datasource")
        raise HTTPException(status_code=403, detail=f"Datasource '{datasource}' not registered")


def format_judge_evaluation(result: dict) -> str:
    """Format judge evaluation history as a collapsible <details> section."""
    verdict = result["verdict"]
    score = result["score"]
    threshold = result.get("threshold", 0.0)
    retries = result["retries"]
    attempts = result.get("attempts", [])

    summary = f"Judge Evaluation: {verdict} ({score:.2f})"

    rows = [
        "| Attempt | Score | Verdict | Feedback |",
        "|---------|-------|---------|----------|",
    ]
    for a in attempts:
        fb = a["feedback"]
        if isinstance(fb, list):
            fb = " ".join(str(x) for x in fb)
        fb = fb.replace("|", "\\|")
        if len(fb) > 200 and fb != "-":
            fb = fb[:197] + "..."
        rows.append(f"| {a['attempt']} | {a['score']:.2f} | {a['verdict']} | {fb} |")

    table = "\n".join(rows)
    footer = f"Threshold: {threshold:.2f} | Retries: {retries}"

    return (
        f"\n\n---\n\n<details>\n<summary>{summary}</summary>\n\n"
        f"{table}\n\n{footer}\n</details>"
    )


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

app = FastAPI(
    title="LangChain Judge API",
    description="LLM-as-a-Judge chain with RAG support",
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "judge-chain"
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = 0.7
    stream: bool = False
    use_rag: bool = Field(default=True, description="Enable RAG context retrieval")


@app.get("/health")
async def health():
    return {"status": "ok"}


def _sse_chunk(run_id: str, content: str, finish_reason: str | None = None) -> str:
    """Format a single SSE data line in OpenAI chat.completion.chunk format."""
    chunk = {
        "id": f"chatcmpl-{run_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "choices": [{
            "index": 0,
            "delta": {"content": content} if content else {},
            "finish_reason": finish_reason,
        }],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _sse_role(run_id: str) -> str:
    """First SSE event: role declaration."""
    chunk = {
        "id": f"chatcmpl-{run_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "choices": [{
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None,
        }],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


async def _stream_response(request: ChatRequest):
    """Async generator that yields SSE events for streaming response."""
    user_message = request.messages[-1].content
    context = ""
    if request.use_rag:
        context = await get_relevant_context(user_message, model_name=request.model)

    run_id = "stream"
    role_sent = False

    try:
        async for event in run_judge_chain_stream(
            prompt=user_message,
            context=context,
            temperature=request.temperature,
        ):
            if not role_sent:
                yield _sse_role(run_id)
                role_sent = True

            if event["type"] == "status":
                yield _sse_chunk(run_id, event["text"])

            elif event["type"] == "token":
                yield _sse_chunk(run_id, event["text"])

            elif event["type"] == "evaluation":
                result = event["result"]
                run_id = result["run_id"]
                details = format_judge_evaluation(result)
                yield _sse_chunk(run_id, details)

        yield _sse_chunk(run_id, "", finish_reason="stop")
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error("Streaming error: %s", e)
        if not role_sent:
            yield _sse_role(run_id)
        yield _sse_chunk(run_id, f"\n\nError during generation: {e}")
        yield _sse_chunk(run_id, "", finish_reason="stop")
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest, raw_request: Request):
    _verify_chat_auth(raw_request)
    if request.stream:
        return StreamingResponse(
            _stream_response(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming path (unchanged from Phase 3D)
    user_message = request.messages[-1].content

    context = ""
    if request.use_rag:
        context = await get_relevant_context(user_message, model_name=request.model)

    result = await run_judge_chain(
        prompt=user_message,
        context=context,
        temperature=request.temperature,
    )

    return {
        "id": f"chatcmpl-{result['run_id']}",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": result["final_answer"] + format_judge_evaluation(result),
            },
            "finish_reason": "stop",
        }],
        "metadata": {
            "judge_verdict": result["verdict"],
            "response_relevancy": result["score"],
            "retries": result["retries"],
            "judge_feedback": result.get("judge_feedback", ""),
        },
    }


# --- Phase 4A: Ingestion endpoints ---


@app.post("/ingest")
async def ingest_upload(
    raw_request: Request,
    file: UploadFile = File(...),
    datasource: str = Form(...),
    dry_run: bool = Form(False),
):
    """Ingest a single file into a datasource collection."""
    _verify_ingest_auth(raw_request)
    _validate_datasource(datasource)

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        if dry_run:
            result = dry_run_file(tmp_path, original_filename=file.filename)
            log_ingest_event("dry_run", datasource, filename=file.filename,
                             chunks=result["total_chunks"])
            return {"dry_run": True, **result}

        chunk_count = ingest_file(tmp_path, datasource, original_filename=file.filename)
        log_ingest_event("ingest", datasource, filename=file.filename, chunks=chunk_count)
        return {
            "status": "ok",
            "filename": file.filename,
            "datasource": datasource,
            "chunks": chunk_count,
        }
    except Exception as e:
        log_ingest_event("ingest", datasource, filename=file.filename,
                         status="error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/ingest/batch")
async def ingest_batch(raw_request: Request, datasource: str = Form(...)):
    """Ingest all supported files in data/documents/{datasource}/."""
    _verify_ingest_auth(raw_request)
    _validate_datasource(datasource)
    try:
        result = ingest_folder(datasource)
        log_ingest_event("batch", datasource, chunks=result["total_chunks"])
        return {"status": "ok", "datasource": datasource, **result}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log_ingest_event("batch", datasource, status="error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/ingest/{datasource}")
async def ingest_delete(datasource: str, raw_request: Request):
    """Delete a datasource collection from ChromaDB."""
    _verify_ingest_auth(raw_request)
    _validate_datasource(datasource)
    try:
        delete_collection(datasource)
        log_ingest_event("delete", datasource)
        return {"status": "ok", "datasource": datasource, "action": "deleted"}
    except Exception as e:
        log_ingest_event("delete", datasource, status="error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingest/{datasource}")
async def ingest_detail(datasource: str, raw_request: Request):
    """List files in a datasource with per-file chunk counts."""
    _verify_ingest_auth(raw_request)
    _validate_datasource(datasource)
    try:
        files = list_files(datasource)
        return {"datasource": datasource, "files": files}
    except chromadb.errors.NotFoundError:
        raise HTTPException(status_code=404, detail=f"Datasource '{datasource}' not found")


@app.delete("/ingest/{datasource}/{filename:path}")
async def ingest_delete_file(datasource: str, filename: str, raw_request: Request):
    """Delete all chunks for a specific file from a datasource."""
    _verify_ingest_auth(raw_request)
    _validate_datasource(datasource)
    try:
        deleted = delete_file(datasource, filename)
        log_ingest_event("delete_file", datasource, filename=filename, chunks=deleted)
        return {"status": "ok", "datasource": datasource, "filename": filename, "deleted_chunks": deleted}
    except chromadb.errors.NotFoundError:
        raise HTTPException(status_code=404, detail=f"Datasource '{datasource}' not found")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# --- Phase 4B: Audit endpoint ---


@app.get("/audit/recent")
async def audit_recent(raw_request: Request, n: int = 20):
    """Return recent audit log entries."""
    _verify_ingest_auth(raw_request)
    events = get_recent_events(n)
    return {"events": events}
