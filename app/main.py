"""
LangChain API Server — OpenAI-compatible endpoint.
Exposes LLM-as-a-Judge chain via /v1/chat/completions.
Phase 3E: SSE streaming support.
Phase 4A: Ingestion endpoints.
"""
import json
import logging
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from chains.llm_as_judge import run_judge_chain, run_judge_chain_stream
from rag.retriever import get_relevant_context
import chromadb.errors
from rag.ingest import (
    ingest_file, ingest_folder, delete_collection,
    list_files, delete_file, SUPPORTED_EXTENSIONS,
)

logger = logging.getLogger(__name__)


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
        context = await get_relevant_context(user_message)

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
async def chat_completions(request: ChatRequest):
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
        context = await get_relevant_context(user_message)

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
    file: UploadFile = File(...),
    datasource: str = Form(...),
):
    """Ingest a single file into a datasource collection."""
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
        chunk_count = ingest_file(tmp_path, datasource, original_filename=file.filename)
        return {
            "status": "ok",
            "filename": file.filename,
            "datasource": datasource,
            "chunks": chunk_count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/ingest/batch")
async def ingest_batch(datasource: str = Form(...)):
    """Ingest all supported files in data/documents/{datasource}/."""
    try:
        result = ingest_folder(datasource)
        return {"status": "ok", "datasource": datasource, **result}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/ingest/{datasource}")
async def ingest_delete(datasource: str):
    """Delete a datasource collection from ChromaDB."""
    try:
        delete_collection(datasource)
        return {"status": "ok", "datasource": datasource, "action": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingest/{datasource}")
async def ingest_detail(datasource: str):
    """List files in a datasource with per-file chunk counts."""
    try:
        files = list_files(datasource)
        return {"datasource": datasource, "files": files}
    except chromadb.errors.NotFoundError:
        raise HTTPException(status_code=404, detail=f"Datasource '{datasource}' not found")


@app.delete("/ingest/{datasource}/{filename:path}")
async def ingest_delete_file(datasource: str, filename: str):
    """Delete all chunks for a specific file from a datasource."""
    try:
        deleted = delete_file(datasource, filename)
        return {"status": "ok", "datasource": datasource, "filename": filename, "deleted_chunks": deleted}
    except chromadb.errors.NotFoundError:
        raise HTTPException(status_code=404, detail=f"Datasource '{datasource}' not found")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
