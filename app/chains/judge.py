"""
Judge LLM chain — evaluates Reasoner output quality.

Phase 3A: Returns verdict (PASS/FAIL), score (1-5), and feedback.
Phase 3B: Verdict/score replaced by RAGAS Response Relevancy.
          evaluate_answer() retained but dormant.
Phase 3C: generate_feedback() active — called on RAGAS FAIL to produce
          actionable feedback for Reasoner retry.
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


# --- Phase 3C: feedback-only generation ---

FEEDBACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a quality improvement advisor.
The following answer was evaluated and scored poorly on Response Relevancy
(how well the answer addresses the original question).

Analyze why the answer may not be relevant and provide specific,
actionable feedback for improvement.

Evaluation criteria:
{criteria}

Respond ONLY in valid JSON:
{{
  "feedback": "specific instructions for improving the answer's relevance to the question"
}}"""),
    ("human", """Question: {question}

Answer that needs improvement:
{answer}

Response Relevancy score: {score} (threshold: {threshold})"""),
])


async def generate_feedback(
    question: str, answer: str, score: float, threshold: float,
    *, criteria: str = "", callback_handler=None,
) -> str:
    """
    Generate improvement feedback using Judge LLM (Phase 3C).
    Called only when RAGAS score indicates FAIL.
    """
    judge = get_judge()
    chain = FEEDBACK_PROMPT | judge | JsonOutputParser()
    cfg = {"callbacks": [callback_handler]} if callback_handler else {}

    try:
        result = await chain.ainvoke({
            "question": question,
            "answer": answer,
            "score": f"{score:.4f}",
            "threshold": f"{threshold:.2f}",
            "criteria": criteria,
        }, config=cfg or None)
        feedback = result.get("feedback", "Please provide a more relevant answer.")
        logger.info("Judge feedback generated (%d chars)", len(feedback))
        return feedback
    except Exception as e:
        logger.error("Judge feedback generation failed: %s", e)
        return "Please provide a more direct and relevant answer to the question."
