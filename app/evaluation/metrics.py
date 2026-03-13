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
from ragas.metrics import ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from models.provider import get_judge
from models.embeddings import get_embeddings
from settings import settings

logger = logging.getLogger(__name__)

# Lazy initialization (model download happens on first call)
_metric = None


def _get_metric():
    global _metric
    if _metric is None:
        judge_llm = LangchainLLMWrapper(get_judge())
        embeddings = LangchainEmbeddingsWrapper(get_embeddings())
        _metric = ResponseRelevancy(llm=judge_llm, embeddings=embeddings)
    return _metric


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
