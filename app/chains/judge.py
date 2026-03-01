"""
Judge LLM chain — evaluates Reasoner output quality.

Phase 3A: Returns verdict (PASS/FAIL), score (1-5), and feedback.
Phase 3B: Verdict/score replaced by RAGAS Response Relevancy.
          This module is NOT called in 3B but retained for 3C (feedback generation).
Phase 3C: Reactivated for feedback-only generation on FAIL.
"""
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from models.provider import get_judge
from settings import settings

logger = logging.getLogger(__name__)

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a strict quality evaluator.
Evaluate the given answer against the original question.

Evaluation criteria:
{criteria}

Respond ONLY in valid JSON:
{{
  "verdict": "PASS" or "FAIL",
  "score": 1 to 5 (integer),
  "feedback": "specific feedback for improvement if FAIL, or brief confirmation if PASS"
}}"""),
    ("human", """Question: {question}

Answer to evaluate:
{answer}"""),
])


async def evaluate_answer(question: str, answer: str) -> dict:
    """
    Evaluate an answer using Judge LLM (Response Relevancy-tentative).

    Returns:
        dict with keys: verdict (str), score (int), feedback (str)
    """
    judge = get_judge()
    chain = JUDGE_PROMPT | judge | JsonOutputParser()

    criteria = settings.judge_criteria

    try:
        result = await chain.ainvoke({
            "criteria": criteria,
            "question": question,
            "answer": answer,
        })
        logger.info(
            "Judge evaluation: verdict=%s score=%s",
            result.get("verdict"),
            result.get("score"),
        )
        return {
            "verdict": result.get("verdict", "PASS"),
            "score": result.get("score", 0),
            "feedback": result.get("feedback", ""),
        }
    except Exception as e:
        logger.error("Judge evaluation failed: %s", e)
        # On judge failure, default to PASS to avoid blocking the user
        return {"verdict": "PASS", "score": 0, "feedback": f"Judge error: {e}"}
