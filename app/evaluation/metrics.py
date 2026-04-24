"""
RAGAS Response Relevancy metric integration.

Uses RAGAS library to compute a quantitative score (0.0-1.0) for how well
the response answers the original question.

Algorithm (RAGAS internal):
1. Judge LLM generates N artificial questions from the response
2. Embedding model computes cosine similarity between original question
   and each generated question
3. Mean similarity = Response Relevancy score

LLM: Judge model via LiteLLM (same model as Phase 3A, now called by RAGAS)
Embeddings: cl-nagoya/ruri-v3-310m (Japanese-optimized, JMTEB top-tier)
"""
import logging
from ragas import SingleTurnSample
from ragas.metrics import ResponseRelevancy, Faithfulness, ContextPrecision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from models.provider import get_judge
from models.embeddings import get_embeddings
from settings import settings

logger = logging.getLogger(__name__)

def _get_metric():
    """Build metric fresh each call (judge endpoint may change after probe)."""
    judge_llm = LangchainLLMWrapper(get_judge())
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    metric = ResponseRelevancy(llm=judge_llm, embeddings=embeddings)
    metric.question_generation.instruction = (
        "Generate a question for the given answer and Identify if answer is noncommittal. "
        "Give noncommittal as 1 if the answer is noncommittal and 0 if the answer is committal. "
        "A noncommittal answer is one that is evasive, vague, or ambiguous. "
        "For example, \"I don't know\" or \"I'm not sure\" are noncommittal answers. "
        "IMPORTANT: Generate the question in the SAME LANGUAGE as the answer."
    )
    return metric


def _get_faithfulness_metric():
    return Faithfulness(llm=LangchainLLMWrapper(get_judge()))


def _get_context_precision_metric():
    return ContextPrecision(llm=LangchainLLMWrapper(get_judge()))


async def compute_response_relevancy(question: str, answer: str) -> dict:
    """
    Compute RAGAS Response Relevancy score.

    Args:
        question: Original user question
        answer: Reasoner's response

    Returns:
        dict with keys:
            score (float): 0.0-1.0
            verdict (str): "PASS" or "FAIL" based on threshold
            threshold (float): configured threshold value
    """
    threshold = settings.ragas_response_relevancy_threshold

    try:
        metric = _get_metric()
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
        )
        score = await metric.single_turn_ascore(sample)

        verdict = "PASS" if score >= threshold else "FAIL"

        logger.info(
            "RAGAS Response Relevancy: score=%.4f threshold=%.2f verdict=%s",
            score,
            threshold,
            verdict,
        )

        return {
            "score": round(float(score), 4),
            "verdict": verdict,
            "threshold": threshold,
        }
    except Exception as e:
        logger.error("RAGAS evaluation failed: %s", e)
        # On RAGAS failure, default to PASS to avoid blocking the user
        return {
            "score": 0.0,
            "verdict": "PASS",
            "threshold": threshold,
        }


async def compute_faithfulness(question: str, context: str, answer: str) -> dict:
    """
    Compute RAGAS Faithfulness score (reference-free, LLM-judge).

    Returns:
        dict with keys: score (float), verdict (str), threshold (float)
    """
    threshold = settings.ragas_faithfulness_threshold
    try:
        metric = _get_faithfulness_metric()
        sample = SingleTurnSample(
            user_input=question,
            retrieved_contexts=[context],
            response=answer,
        )
        score = await metric.single_turn_ascore(sample)
        verdict = "PASS" if score >= threshold else "FAIL"
        logger.info(
            "RAGAS Faithfulness: score=%.4f threshold=%.2f verdict=%s",
            score, threshold, verdict,
        )
        return {"score": round(float(score), 4), "verdict": verdict, "threshold": threshold}
    except Exception as e:
        logger.error("RAGAS Faithfulness evaluation failed: %s", e)
        return {"score": 0.0, "verdict": "PASS", "threshold": threshold}


async def compute_context_precision(
    question: str, context: str, answer: str, *, reference: str = ""
) -> dict:
    """
    Compute RAGAS Context Precision score (reference-based, LLM-judge).

    Requires reference (expected answer). Returns {"score": None} when reference is empty.

    Returns:
        dict with keys: score (float | None), verdict (str), threshold (float)
    """
    threshold = settings.ragas_context_precision_threshold
    if not reference:
        return {"score": None, "verdict": "SKIP", "threshold": threshold}
    try:
        metric = _get_context_precision_metric()
        sample = SingleTurnSample(
            user_input=question,
            retrieved_contexts=[context],
            response=answer,
            reference=reference,
        )
        score = await metric.single_turn_ascore(sample)
        verdict = "PASS" if score >= threshold else "FAIL"
        logger.info(
            "RAGAS ContextPrecision: score=%.4f threshold=%.2f verdict=%s",
            score, threshold, verdict,
        )
        return {"score": round(float(score), 4), "verdict": verdict, "threshold": threshold}
    except Exception as e:
        logger.error("RAGAS ContextPrecision evaluation failed: %s", e)
        return {"score": 0.0, "verdict": "PASS", "threshold": threshold}
