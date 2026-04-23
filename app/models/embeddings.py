"""
Shared embedding model singletons, one per role.

- role="query": always CPU (avoids VRAM contention with the Reasoner during inference)
- role="ingest": uses settings.ingest_device (default "cpu"; set "cuda" for batch
  ingestion on a GPU host)

Prefix conventions are model-specific and live in models.embedding_adapters.
"""
from typing import Literal
from langchain_community.embeddings import HuggingFaceEmbeddings
from settings import settings

_query_embeddings = None
_ingest_embeddings = None


def get_embeddings(role: Literal["query", "ingest"] = "query") -> HuggingFaceEmbeddings:
    global _query_embeddings, _ingest_embeddings
    if role == "query":
        if _query_embeddings is None:
            _query_embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return _query_embeddings
    else:
        if _ingest_embeddings is None:
            _ingest_embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model_name,
                model_kwargs={"device": settings.ingest_device},
                encode_kwargs={"normalize_embeddings": True},
            )
        return _ingest_embeddings
