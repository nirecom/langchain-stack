"""
ChromaDB retriever for RAG context.
Phase 3: Returns empty string (no RAG DB available).
Phase 4: Will query ChromaDB for relevant document chunks.
"""
import logging

logger = logging.getLogger(__name__)


async def get_relevant_context(query: str, n_results: int = 3) -> str:
    """
    Retrieve relevant document chunks from ChromaDB.

    Phase 3: Always returns empty string (RAG not yet implemented).
    Phase 4: Will connect to ChromaDB and perform vector search.
    """
    # TODO(Phase 4): Implement ChromaDB retrieval
    return ""
