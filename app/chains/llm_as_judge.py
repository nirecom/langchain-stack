"""
LLM-as-a-Judge orchestration: RAGAS scoring, Judge feedback, and retry loop.

Phase 3B: Reasoner → RAGAS Response Relevancy (no retry).
Phase 3C: RAGAS score → FAIL → Judge LLM feedback → Reasoner retry → RAGAS re-score.
Phase 3E: SSE streaming variant (run_judge_chain_stream).
"""
import uuid
import logging
from collections.abc import AsyncGenerator
from langchain_core.messages import SystemMessage, HumanMessage
from models.provider import get_reasoner, probe_endpoints
from evaluation.metrics import compute_response_relevancy
from chains.judge import generate_feedback
from settings import settings as app_settings

logger = logging.getLogger(__name__)

REASONER_SYSTEM_PROMPT = "Respond in the same language as the user's question."


def _build_reasoner_messages(reasoner_input: str) -> list:
    return [
        SystemMessage(content=REASONER_SYSTEM_PROMPT),
        HumanMessage(content=reasoner_input),
    ]


def _build_reasoner_input(
    prompt: str,
    context: str,
    attempt: int,
    evaluation: dict,
    feedback: str,
) -> str:
    """Build the reasoner prompt, incorporating feedback on retry attempts."""
    if attempt == 0:
        return (
            f"Context:\n{context}\n\nQuestion: {prompt}" if context else prompt
        )
    return (
        (f"Context:\n{context}\n\n" if context else "")
        + f"Question: {prompt}\n\n"
        + f"Your previous answer was rejected (Response Relevancy score: "
        + f"{evaluation.get('score', 0):.4f}, threshold: "
        + f"{evaluation.get('threshold', 0):.2f}).\n"
        + f"Feedback: {feedback}\n"
        + f"Please provide an improved answer."
    )


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
    await probe_endpoints()
    max_retries = max_retries if max_retries is not None else app_settings.max_judge_retries

    run_id = str(uuid.uuid4())[:8]
    reasoner = get_reasoner(temperature=temperature)

    answer = None
    evaluation = {}
    feedback = ""
    attempts_history = []

    for attempt in range(max_retries + 1):
        reasoner_input = _build_reasoner_input(
            prompt, context, attempt, evaluation, feedback
        )

        answer_msg = await reasoner.ainvoke(_build_reasoner_messages(reasoner_input))
        answer = answer_msg.content

        logger.info(
            "[%s] attempt=%d Reasoner generated answer (%d chars)",
            run_id, attempt, len(answer),
        )

        # RAGAS Response Relevancy evaluation
        evaluation = await compute_response_relevancy(
            question=prompt, answer=answer
        )

        logger.info(
            "[%s] attempt=%d RAGAS score=%.4f verdict=%s",
            run_id, attempt, evaluation["score"], evaluation["verdict"],
        )

        if evaluation["verdict"] == "PASS":
            attempts_history.append({
                "attempt": attempt + 1,
                "score": evaluation["score"],
                "verdict": "PASS",
                "feedback": "-",
            })
            break

        # On FAIL, generate feedback (skip on last attempt)
        if attempt < max_retries:
            criteria = (
                app_settings.rag_judge_criteria if context
                else app_settings.judge_criteria
            )
            feedback = await generate_feedback(
                question=prompt,
                answer=answer,
                score=evaluation["score"],
                threshold=evaluation["threshold"],
                criteria=criteria,
            )
            attempts_history.append({
                "attempt": attempt + 1,
                "score": evaluation["score"],
                "verdict": "FAIL",
                "feedback": feedback,
            })
            logger.info(
                "[%s] attempt=%d FAIL — retrying with feedback",
                run_id, attempt,
            )
        else:
            attempts_history.append({
                "attempt": attempt + 1,
                "score": evaluation["score"],
                "verdict": "FAIL",
                "feedback": "-",
            })

    return {
        "run_id": run_id,
        "final_answer": answer,
        "verdict": evaluation.get("verdict", "PASS"),
        "score": evaluation.get("score", 0.0),
        "retries": attempt,
        "judge_feedback": feedback,
        "attempts": attempts_history,
        "threshold": evaluation.get("threshold", 0.0),
    }


async def run_judge_chain_stream(
    prompt: str,
    context: str = "",
    temperature: float = 0.7,
    max_retries: int | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Streaming variant of run_judge_chain.

    Yields event dicts:
        {"type": "status", "text": "..."}      — evaluation progress
        {"type": "token",  "text": "..."}      — answer tokens
        {"type": "evaluation", "result": {...}} — final metadata
    """
    await probe_endpoints()
    max_retries = max_retries if max_retries is not None else app_settings.max_judge_retries

    run_id = str(uuid.uuid4())[:8]
    reasoner = get_reasoner(temperature=temperature)

    answer = None
    evaluation = {}
    feedback = ""
    attempts_history = []

    for attempt in range(max_retries + 1):
        is_last_attempt = attempt == max_retries

        reasoner_input = _build_reasoner_input(
            prompt, context, attempt, evaluation, feedback
        )

        yield {"type": "status", "text": f"⏳ Generating answer (attempt {attempt + 1})...\n\n"}

        if is_last_attempt:
            # Last attempt: stream tokens since this answer will be used regardless
            answer_chunks = []
            async for chunk in reasoner.astream(_build_reasoner_messages(reasoner_input)):
                token = chunk.content
                if token:
                    answer_chunks.append(token)
                    yield {"type": "token", "text": token}
            answer = "".join(answer_chunks)
        else:
            # Non-final: use ainvoke (need RAGAS verdict before committing)
            answer_msg = await reasoner.ainvoke(_build_reasoner_messages(reasoner_input))
            answer = answer_msg.content

        logger.info(
            "[%s] attempt=%d Reasoner generated answer (%d chars)",
            run_id, attempt, len(answer),
        )

        # RAGAS evaluation (always non-streamed)
        yield {"type": "status", "text": "⏳ Evaluating response relevancy...\n\n"}
        evaluation = await compute_response_relevancy(
            question=prompt, answer=answer
        )

        logger.info(
            "[%s] attempt=%d RAGAS score=%.4f verdict=%s",
            run_id, attempt, evaluation["score"], evaluation["verdict"],
        )

        if evaluation["verdict"] == "PASS":
            attempts_history.append({
                "attempt": attempt + 1,
                "score": evaluation["score"],
                "verdict": "PASS",
                "feedback": "-",
            })
            if not is_last_attempt:
                # Answer was not streamed yet — emit it now
                yield {"type": "token", "text": answer}
            break

        # FAIL path
        if not is_last_attempt:
            yield {
                "type": "status",
                "text": (
                    f"⏳ Response Relevancy: {evaluation['score']:.2f}"
                    f" — FAIL (threshold: {evaluation['threshold']:.2f})\n"
                    f"⏳ Generating feedback and retrying...\n\n"
                ),
            }
            criteria = (
                app_settings.rag_judge_criteria if context
                else app_settings.judge_criteria
            )
            feedback = await generate_feedback(
                question=prompt,
                answer=answer,
                score=evaluation["score"],
                threshold=evaluation["threshold"],
                criteria=criteria,
            )
            attempts_history.append({
                "attempt": attempt + 1,
                "score": evaluation["score"],
                "verdict": "FAIL",
                "feedback": feedback,
            })
            logger.info(
                "[%s] attempt=%d FAIL — retrying with feedback",
                run_id, attempt,
            )
        else:
            yield {
                "type": "status",
                "text": (
                    f"⚠️ Response Relevancy: {evaluation['score']:.2f}"
                    f" — FAIL (max retries reached)\n\n"
                ),
            }
            attempts_history.append({
                "attempt": attempt + 1,
                "score": evaluation["score"],
                "verdict": "FAIL",
                "feedback": "-",
            })

    # Final metadata
    yield {
        "type": "evaluation",
        "result": {
            "run_id": run_id,
            "final_answer": answer,
            "verdict": evaluation.get("verdict", "PASS"),
            "score": evaluation.get("score", 0.0),
            "retries": attempt,
            "judge_feedback": feedback,
            "attempts": attempts_history,
            "threshold": evaluation.get("threshold", 0.0),
        },
    }
