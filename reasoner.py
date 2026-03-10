"""
Reasoner chain — generates answers to user questions.
Supports initial generation and retry with Judge feedback.
"""
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI


async def generate_answer(
    llm: ChatOpenAI,
    question: str,
    context: str = "",
    feedback: str = "",
) -> str:
    """
    Generate an answer using the Reasoner LLM.

    Args:
        llm: The Reasoner LLM instance.
        question: The user's question.
        context: Optional RAG context (Phase 4).
        feedback: Optional Judge feedback for retry attempts.

    Returns:
        The generated answer text.
    """
    parts = []

    if context:
        parts.append(f"Context:\n{context}\n")

    parts.append(f"Question: {question}")

    if feedback:
        parts.append(
            f"\nYour previous answer was rejected. Feedback: {feedback}\n"
            f"Please provide an improved answer."
        )

    prompt = "\n".join(parts)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content
