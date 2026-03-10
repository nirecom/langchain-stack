"""
LLM-as-a-Judge orchestration chain (hybrid mode).

Flow:
  1. Reasoner LLM generates an answer
  2. RAGAS Response Relevancy computes a quantitative score (0-1)
  3. If score >= threshold → PASS, return answer
  4. If score < threshold → FAIL:
     a. Judge LLM generates specific feedback
     b. Reasoner retries with feedback (up to max_retries)
  5. After max retries, return the best-scoring answer

PASS/FAIL is determined SOLELY by the RAGAS score.
Judge LLM is only used for feedback text generation on FAIL.
"""
import logging
import uuid

from evaluation.metrics import compute_response_relevancy
from chains.reasoner import generate_answer
from chains.judge import generate_feedback
from models.provider import get_reasoner, get_judge
from settings import settings

logger = logging.getLogger(__name__)


async def run_judge_chain(
    prompt: str,
    context: str = "",
    temperature: float = 0.7,
    max_retries: int | None = None,
    profile: str = "default",
) -> dict:
    """
    Execute the full LLM-as-a-Judge chain with RAGAS scoring.

    Args:
        prompt: The user's question.
        context: Optional RAG context (Phase 4).
        temperature: Reasoner temperature.
        max_retries: Override for max retry count.
        profile: Judge rules profile ("default" or "rag").

    Returns:
        Dict with final_answer, verdict, response_relevancy score,
        retry count, and judge feedback.
    """
    max_retries = max_retries if max_retries is not None else settings.max_judge_retries
    threshold = settings.get_pass_threshold(profile)
    criteria = settings.get_feedback_criteria(profile)

    reasoner = get_reasoner(temperature=temperature)
    judge = get_judge()

    best_answer = ""
    best_score = -1.0
    feedback = ""
    final_verdict = "FAIL"

    for attempt in range(max_retries + 1):
        # Step 1: Generate answer
        answer = await generate_answer(
            llm=reasoner,
            question=prompt,
            context=context,
            feedback=feedback if attempt > 0 else "",
        )

        # Step 2: Compute RAGAS Response Relevancy score
        score = await compute_response_relevancy(
            user_input=prompt,
            response=answer,
        )

        logger.info(
            "Attempt %d/%d: Response Relevancy=%.3f (threshold=%.3f)",
            attempt + 1,
            max_retries + 1,
            score,
            threshold,
        )

        # Track the best answer across all attempts
        if score > best_score:
            best_score = score
            best_answer = answer

        # Step 3: PASS/FAIL decision based purely on RAGAS score
        if score >= threshold:
            final_verdict = "PASS"
            break

        # Step 4: On FAIL, generate feedback for retry (skip on last attempt)
        if attempt < max_retries:
            feedback = await generate_feedback(
                llm=judge,
                question=prompt,
                answer=answer,
                score=score,
                threshold=threshold,
                criteria=criteria,
            )

    return {
        "run_id": str(uuid.uuid4())[:8],
        "final_answer": best_answer,
        "verdict": final_verdict,
        "response_relevancy": round(best_score, 4),
        "retries": attempt,
        "judge_feedback": feedback if final_verdict == "FAIL" else "",
    }
