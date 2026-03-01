"""
LLM-as-a-Judge orchestration: RAGAS scoring, Judge feedback, and retry loop.

Phase 3B: Reasoner → RAGAS Response Relevancy (no retry).
Phase 3C: RAGAS score → FAIL → Judge LLM feedback → Reasoner retry → RAGAS re-score.
"""
import uuid
import logging
from models.provider import get_reasoner
from evaluation.metrics import compute_response_relevancy
from chains.judge import generate_feedback
from settings import settings as app_settings

logger = logging.getLogger(__name__)


async def run_judge_chain(
    prompt: str,
    context: str = "",
    temperature: float = 0.7,
    max_retries: int | None = None,
) -> dict:
    """
    Run Reasoner → RAGAS → (FAIL → Judge feedback → Retry) pipeline.

    Args:
        prompt: User question
        context: RAG context (unused in Phase 3, reserved for Phase 4)
        temperature: Reasoner temperature (fixed across retries)
        max_retries: Override for max retry count (default from settings)

    Returns:
        dict with final_answer, verdict, score, retries, judge_feedback
    """
    max_retries = max_retries if max_retries is not None else app_settings.max_judge_retries

    run_id = str(uuid.uuid4())[:8]
    reasoner = get_reasoner(temperature=temperature)

    answer = None
    evaluation = {}
    feedback = ""

    for attempt in range(max_retries + 1):
        # Step 1: Generate answer (with feedback if retrying)
        if attempt == 0:
            reasoner_input = (
                f"Context:\n{context}\n\nQuestion: {prompt}" if context else prompt
            )
        else:
            reasoner_input = (
                (f"Context:\n{context}\n\n" if context else "")
                + f"Question: {prompt}\n\n"
                + f"Your previous answer was rejected (Response Relevancy score: "
                + f"{evaluation.get('score', 0):.4f}, threshold: "
                + f"{evaluation.get('threshold', 0):.2f}).\n"
                + f"Feedback: {feedback}\n"
                + f"Please provide an improved answer."
            )

        answer_msg = await reasoner.ainvoke(reasoner_input)
        answer = answer_msg.content

        logger.info(
            "[%s] attempt=%d Reasoner generated answer (%d chars)",
            run_id, attempt, len(answer),
        )

        # Step 2: RAGAS Response Relevancy evaluation
        evaluation = await compute_response_relevancy(
            question=prompt, answer=answer
        )

        logger.info(
            "[%s] attempt=%d RAGAS score=%.4f verdict=%s",
            run_id, attempt, evaluation["score"], evaluation["verdict"],
        )

        if evaluation["verdict"] == "PASS":
            break

        # Step 3: On FAIL, generate feedback (skip on last attempt)
        if attempt < max_retries:
            feedback = await generate_feedback(
                question=prompt,
                answer=answer,
                score=evaluation["score"],
                threshold=evaluation["threshold"],
            )
            logger.info(
                "[%s] attempt=%d FAIL — retrying with feedback",
                run_id, attempt,
            )

    return {
        "run_id": run_id,
        "final_answer": answer,
        "verdict": evaluation.get("verdict", "PASS"),
        "score": evaluation.get("score", 0.0),
        "retries": attempt,
        "judge_feedback": feedback,
    }
