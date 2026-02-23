"""
LangChain API Server — OpenAI-compatible endpoint.
Exposes LLM-as-a-Judge chain via /v1/chat/completions.
"""
from fastapi import FastAPI
from pydantic import BaseModel, Field
from chains.llm_as_judge import run_judge_chain
from rag.retriever import get_relevant_context

app = FastAPI(
    title="LangChain Judge API",
    description="LLM-as-a-Judge chain with RAG support",
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "judge-chain"
    messages: list[ChatMessage]
    temperature: float = 0.7
    use_rag: bool = Field(default=True, description="Enable RAG context retrieval")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    user_message = request.messages[-1].content

    # RAG retrieval (Phase 4 — currently returns empty string)
    context = ""
    if request.use_rag:
        context = await get_relevant_context(user_message)

    # Run LLM-as-a-Judge chain
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
                "content": result["final_answer"],
            },
            "finish_reason": "stop",
        }],
        "metadata": {
            "judge_verdict": result["verdict"],
            "judge_score": result["score"],
            "retries": result["retries"],
        },
    }
