"""
LLM-as-a-Judge orchestration: RAGAS scoring, Judge feedback, and retry loop.

Phase 3B: Reasoner → RAGAS Response Relevancy (no retry).
Phase 3C: RAGAS score → FAIL → Judge LLM feedback → Reasoner retry → RAGAS re-score.
Phase 3E: SSE streaming variant (run_judge_chain_stream).
Phase 5a: Langfuse tracing via app/tracing.py (RAG retrieval moved inside chain).
"""
import uuid
import logging
from collections.abc import AsyncGenerator
from datetime import date

from langchain_core.messages import SystemMessage, HumanMessage
from models.provider import get_reasoner, probe_endpoints
from evaluation.metrics import compute_response_relevancy
from chains.judge import generate_feedback
from rag.retriever import get_relevant_context
from rag.access_control import get_permitted_datasources_for_user
from settings import settings as app_settings
from tracing import trace_span, get_callback_handler

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


def _build_trace_attrs(
    run_id: str,
    user: str | None,
    datasources: list[str],
    reasoner_model: str,
    max_retries: int,
) -> dict:
    username = user or "anonymous"
    session_id = f"{username}:{date.today().isoformat()}"
    return {
        "user_id": username,
        "session_id": session_id,
        "metadata": {
            "run_id": run_id,
            "datasources": datasources,
            "embedding_model": app_settings.embedding_model_name,
            "search_mode": app_settings.search_mode,
            "top_k": app_settings.rag_top_k,
            "reasoner_model": reasoner_model,
            "max_retries": max_retries,
            "ragas_threshold": app_settings.ragas_response_relevancy_threshold,
        },
    }


async def run_judge_chain(
    prompt: str,
    *,
    user: str | None = None,
    use_rag: bool = True,
    temperature: float = 0.7,
    max_retries: int | None = None,
) -> dict:
    """
    Run Reasoner → RAGAS → (FAIL → Judge feedback → Retry) pipeline.

    Args:
        prompt: User question
        user: Authenticated username (sets Langfuse user_id and scopes RAG ACL)
        use_rag: When True, retrieves context from OpenSearch before reasoning
        temperature: Reasoner temperature (fixed across retries)
        max_retries: Override for max retry count (default from settings)

    Returns:
        dict with final_answer, verdict, score, retries, judge_feedback
    """
    await probe_endpoints()
    max_retries = max_retries if max_retries is not None else app_settings.max_judge_retries

    run_id = str(uuid.uuid4())[:8]
    reasoner = get_reasoner(temperature=temperature)
    reasoner_model = getattr(reasoner, "model_name", "") or ""

    datasources = sorted(get_permitted_datasources_for_user(user)) if user else []
    trace_attrs = _build_trace_attrs(run_id, user, datasources, reasoner_model, max_retries)
    tags = ["judge-chain"]

    answer = None
    evaluation: dict = {}
    feedback = ""
    attempts_history: list = []
    attempt = 0

    root_cm = trace_span(
        "judge_chain",
        input={"prompt": prompt},
        as_root=True,
        trace_attrs={**trace_attrs, "tags": tags},
    )
    root = root_cm.__enter__()
    try:
        with trace_span("rag_retrieval", input={"query": prompt}) as rag_span:
            context = await get_relevant_context(prompt, user=user) if use_rag else ""
            tags.append("rag" if context else "no-rag")
            rag_span.update(output={"context": context, "context_chars": len(context)})

        for attempt in range(max_retries + 1):
            reasoner_input = _build_reasoner_input(
                prompt, context, attempt, evaluation, feedback
            )

            with trace_span(f"attempt_{attempt}"):
                cb = get_callback_handler()
                cfg = {"callbacks": [cb]} if cb else {}

                answer_msg = await reasoner.ainvoke(
                    _build_reasoner_messages(reasoner_input), config=cfg or None
                )
                answer = answer_msg.content

                logger.info(
                    "[%s] attempt=%d Reasoner generated answer (%d chars)",
                    run_id, attempt, len(answer),
                )

                with trace_span("ragas_eval") as eval_span:
                    ragas_cb = get_callback_handler()
                    evaluation = await compute_response_relevancy(
                        question=prompt, answer=answer, callback_handler=ragas_cb
                    )
                    eval_span.update(output=evaluation)

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
                        callback_handler=cb,
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

        root.update(
            output={
                "final_answer": answer,
                "verdict": evaluation.get("verdict", "PASS"),
                "score": evaluation.get("score", 0.0),
                "retries": attempt,
                "threshold": evaluation.get("threshold", 0.0),
            },
            metadata={"attempts": attempts_history},
        )
    except BaseException as exc:
        root.update(
            output={
                "error": type(exc).__name__,
                "partial_answer": answer,
            },
            metadata={"attempts": attempts_history},
            level="ERROR",
        )
        raise
    finally:
        root_cm.__exit__(None, None, None)

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
    *,
    user: str | None = None,
    use_rag: bool = True,
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
    import asyncio

    await probe_endpoints()
    max_retries = max_retries if max_retries is not None else app_settings.max_judge_retries

    run_id = str(uuid.uuid4())[:8]
    reasoner = get_reasoner(temperature=temperature)
    reasoner_model = getattr(reasoner, "model_name", "") or ""

    datasources = sorted(get_permitted_datasources_for_user(user)) if user else []
    trace_attrs = _build_trace_attrs(run_id, user, datasources, reasoner_model, max_retries)
    tags = ["judge-chain"]

    answer = None
    evaluation: dict = {}
    feedback = ""
    attempts_history: list = []
    attempt = 0

    root_cm = trace_span(
        "judge_chain",
        input={"prompt": prompt},
        as_root=True,
        trace_attrs={**trace_attrs, "tags": tags},
    )
    root = root_cm.__enter__()
    try:
        with trace_span("rag_retrieval", input={"query": prompt}) as rag_span:
            context = await get_relevant_context(prompt, user=user) if use_rag else ""
            tags.append("rag" if context else "no-rag")
            rag_span.update(output={"context": context, "context_chars": len(context)})

        for attempt in range(max_retries + 1):
            is_last_attempt = attempt == max_retries

            reasoner_input = _build_reasoner_input(
                prompt, context, attempt, evaluation, feedback
            )

            yield {"type": "status", "text": f"⏳ Generating answer (attempt {attempt + 1})...\n\n"}

            with trace_span(f"attempt_{attempt}"):
                cb = get_callback_handler()
                cfg = {"callbacks": [cb]} if cb else {}

                if is_last_attempt:
                    answer_chunks = []
                    async for chunk in reasoner.astream(
                        _build_reasoner_messages(reasoner_input), config=cfg or None
                    ):
                        token = chunk.content
                        if token:
                            answer_chunks.append(token)
                            yield {"type": "token", "text": token}
                    answer = "".join(answer_chunks)
                else:
                    answer_msg = await reasoner.ainvoke(
                        _build_reasoner_messages(reasoner_input), config=cfg or None
                    )
                    answer = answer_msg.content

                logger.info(
                    "[%s] attempt=%d Reasoner generated answer (%d chars)",
                    run_id, attempt, len(answer),
                )

                yield {"type": "status", "text": "⏳ Evaluating response relevancy...\n\n"}
                with trace_span("ragas_eval") as eval_span:
                    ragas_cb = get_callback_handler()
                    evaluation = await compute_response_relevancy(
                        question=prompt, answer=answer, callback_handler=ragas_cb
                    )
                    eval_span.update(output=evaluation)

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
                        yield {"type": "answer", "text": answer}
                    break

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
                        callback_handler=cb,
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

        root.update(
            output={
                "final_answer": answer,
                "verdict": evaluation.get("verdict", "PASS"),
                "score": evaluation.get("score", 0.0),
                "retries": attempt,
                "threshold": evaluation.get("threshold", 0.0),
            },
            metadata={"attempts": attempts_history},
        )
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
    except (asyncio.CancelledError, GeneratorExit) as exc:
        root.update(
            output={"error": type(exc).__name__, "partial_answer": answer},
            metadata={"attempts": attempts_history},
            level="WARNING",
        )
        raise
    except BaseException as exc:
        root.update(
            output={"error": type(exc).__name__, "partial_answer": answer},
            metadata={"attempts": attempts_history},
            level="ERROR",
        )
        raise
    finally:
        root_cm.__exit__(None, None, None)
