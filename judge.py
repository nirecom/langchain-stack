"""
Judge feedback chain — generates improvement guidance when RAGAS score is FAIL.

This chain is ONLY invoked when the RAGAS Response Relevancy score falls
below the pass threshold. It does NOT make the PASS/FAIL decision — that
is determined purely by the RAGAS quantitative score.
"""
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

FEEDBACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a quality evaluator. The following answer was scored
poorly on Response Relevancy — it does not adequately address the question.

Evaluation criteria:
{criteria}

Your job is to provide SPECIFIC, ACTIONABLE feedback so the answer can be
improved. Focus on what is missing or irrelevant.

Respond in plain text (not JSON). Keep feedback under 150 words."""),
    ("human", """Question: {question}
Answer to evaluate: {answer}
Response Relevancy score: {score:.3f} (threshold: {threshold:.3f})"""),
])


async def generate_feedback(
    llm: ChatOpenAI,
    question: str,
    answer: str,
    score: float,
    threshold: float,
    criteria: str,
) -> str:
    """
    Generate improvement feedback for a low-scoring answer.

    Args:
        llm: The Judge LLM instance.
        question: The original user question.
        answer: The answer that scored below threshold.
        score: The RAGAS Response Relevancy score (0-1).
        threshold: The pass threshold from judge_rules.yaml.
        criteria: Formatted feedback criteria string.

    Returns:
        Feedback text for the Reasoner to use in retry.
    """
    try:
        chain = FEEDBACK_PROMPT | llm
        response = await chain.ainvoke({
            "criteria": criteria,
            "question": question,
            "answer": answer,
            "score": score,
            "threshold": threshold,
        })
        feedback = response.content.strip()
        logger.info("Judge feedback generated (%d chars)", len(feedback))
        return feedback
    except Exception:
        logger.exception("Failed to generate Judge feedback")
        # Return generic feedback so the retry loop can still proceed
        return (
            "The answer does not sufficiently address the question. "
            "Please provide a more focused and complete response."
        )
