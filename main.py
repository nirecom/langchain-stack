"""
LangChain Judge API Server — OpenAI-compatible endpoint.
Exposes LLM-as-a-Judge chain via /v1/chat/completions.

Phase 3: RAGAS Response Relevancy scoring with hybrid feedback.
"""
import logging

from fastapi import FastAPI
from pydantic import BaseModel, Field

from chains.llm_as_judge import run_judge_chain
from rag.retriever import get_relevant_context

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

app = FastAPI(
    title="LangChain Judge API",
    description="LLM-as-a-Judge chain with RAGAS Response Relevancy scoring",
    version="0.3.0",
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "judge-chain"
    messages: list[ChatMessage]
    temperature: float = 0.7
    use_rag: bool = Field(
        default=False,
        description="Enable RAG context retrieval (Phase 4)",
    )


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.3.0"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    user_message = request.messages[-1].content

    # RAG retrieval (Phase 4 — currently returns empty)
    context = ""
    if request.use_rag:
        context = await get_relevant_context(user_message)

    # Select evaluation profile based on context availability
    profile = "rag" if context else "default"

    # Run LLM-as-a-Judge chain with RAGAS scoring
    result = await run_judge_chain(
        prompt=user_message,
        context=context,
        temperature=request.temperature,
        profile=profile,
    )

    return {
        "id": f"chatcmpl-{result['run_id']}",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": result["final_answer"],
            },
            "finish_reason": "stop",
        }],
        "metadata": {
            "judge_verdict": result["verdict"],
            "response_relevancy": result["response_relevancy"],
            "retries": result["retries"],
            "judge_feedback": result["judge_feedback"],
        },
    }
