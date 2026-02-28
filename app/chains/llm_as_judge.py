"""
LLM-as-a-Judge orchestration.

Phase 3A: Reasoner → Judge LLM (no retry).
Phase 3B: Judge LLM verdict replaced by RAGAS score.
Phase 3C: Retry loop with feedback added.
"""
import uuid
import logging
from models.provider import get_reasoner
from chains.judge import evaluate_answer

logger = logging.getLogger(__name__)


async def run_judge_chain(
    prompt: str,
    context: str = "",
    temperature: float = 0.7,
) -> dict:
    """
    Run Reasoner → Judge pipeline (no retry in Phase 3A).

    Args:
        prompt: User question
        context: RAG context (unused in Phase 3, reserved for Phase 4)
        temperature: Reasoner temperature

    Returns:
        dict with final_answer, verdict, score, feedback, retries
    """
    run_id = str(uuid.uuid4())[:8]
    reasoner = get_reasoner(temperature=temperature)

    # Step 1: Generate answer
    reasoner_input = (
        f"Context:\n{context}\n\nQuestion: {prompt}" if context else prompt
    )
    answer_msg = await reasoner.ainvoke(reasoner_input)
    answer = answer_msg.content

    logger.info("[%s] Reasoner generated answer (%d chars)", run_id, len(answer))

    # Step 2: Judge the answer
    evaluation = await evaluate_answer(question=prompt, answer=answer)

    logger.info(
        "[%s] Judge result: verdict=%s score=%s",
        run_id,
        evaluation["verdict"],
        evaluation["score"],
    )

    return {
        "run_id": run_id,
        "final_answer": answer,
        "verdict": evaluation["verdict"],
        "score": evaluation["score"],
        "retries": 0,
        "judge_feedback": evaluation["feedback"],
    }
