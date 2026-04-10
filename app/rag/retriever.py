"""
ChromaDB retriever with ACL-scoped multi-collection search.

Phase 4C: Queries permitted collections per model, merges results by distance.
n_results is the final total hit count across all collections — each permitted
collection is queried for up to n_results candidates, then the merged pool is
truncated to n_results by ascending distance. Ties are broken by permitted-
collection alphabetical order (stable sort).
"""
import logging

import chromadb.errors

from models.chroma import get_chroma_client
from models.embeddings import get_embeddings, QUERY_PREFIX
from rag.access_control import get_permitted_datasources
from rag.audit import log_retrieve_event
from settings import settings

logger = logging.getLogger(__name__)

# Matches the prefix used by ingest.py when storing documents.
DOCUMENT_PREFIX = "検索文書: "


async def get_relevant_context(
    query: str,
    *,
    model_name: str = "",
    n_results: int | None = None,
) -> str:
    """
    Retrieve RAG context for *query*, restricted to ACL-permitted collections
    for *model_name*. Returns "" on failure or when no results are found.
    """
    if not query.strip():
        return ""

    k = n_results if n_results is not None else settings.rag_top_k

    permitted = get_permitted_datasources(model_name)
    if not permitted:
        logger.warning(
            "RAG: no datasources permitted for model '%s'", model_name,
        )
        log_retrieve_event(model_name, [], query, 0,
                           status="no_permitted_datasources")
        return ""

    # Use embed_documents (not embed_query) to avoid internal prefix handling
    qvec = get_embeddings().embed_documents([QUERY_PREFIX + query])[0]

    client = get_chroma_client()

    # ruri-v3-310m embeddings are L2-normalized (see models/embeddings.py
    # encode_kwargs={"normalize_embeddings": True}). ChromaDB uses L2 by
    # default. For unit vectors, L2² = 2 - 2·cos, so ordering by L2 is
    # monotone-equivalent to ordering by cosine similarity.
    # Cross-collection merge by raw L2 distance is therefore valid.
    candidates: list[tuple[float, int, str]] = []  # (distance, col_idx, text)

    for idx, ds in enumerate(sorted(permitted)):
        try:
            col = client.get_collection(name=ds)
        except chromadb.errors.NotFoundError:
            logger.warning("RAG: collection '%s' not found — skipping", ds)
            continue

        res = col.query(
            query_embeddings=[qvec],
            n_results=k,
            include=["documents", "distances"],
        )
        docs = (res.get("documents") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        for doc, dist in zip(docs, dists):
            clean = doc.removeprefix(DOCUMENT_PREFIX)
            candidates.append((dist, idx, clean))

    candidates.sort(key=lambda t: (t[0], t[1]))
    top = candidates[:k]

    log_retrieve_event(model_name, sorted(permitted), query, len(top))

    return "\n\n---\n\n".join(text for _, _, text in top) if top else ""
