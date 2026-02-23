"""
LLM-as-a-Judge orchestration chain.

Phase 2 (current): Reasoner passthrough only.
Phase 3 (TODO): Add Judge evaluation with retry loop.
"""
import uuid
from models.provider import get_reasoner


async def run_judge_chain(
    prompt: str,
    context: str = "",
    temperature: float = 0.7,
    max_retries: int | None = None,
) -> dict:
    """
    Run the LLM-as-a-Judge chain.

    Phase 2: Calls reasoner directly, returns answer without judge evaluation.
    Phase 3: Will add Judge LLM evaluation and retry loop.
    """
    reasoner = get_reasoner(temperature=temperature)

    # Build input with context if provided
    reasoner_input = (
        f"Context:\n{context}\n\nQuestion: {prompt}"
        if context else prompt
    )

    answer = await reasoner.ainvoke(reasoner_input)

    return {
        "run_id": str(uuid.uuid4())[:8],
        "final_answer": answer.content,
        "verdict": "PASS",       # Phase 2: always PASS (no judge yet)
        "score": None,           # Phase 3: judge will provide score
        "retries": 0,
        "judge_feedback": "",
    }
