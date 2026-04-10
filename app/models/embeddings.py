"""
Shared embedding model singleton.

ruri-v3-310m is used by both RAGAS evaluation (metrics.py) and
document ingestion (ingest.py). Prefix convention (検索文書/検索クエリ)
is applied by callers, not here.
"""

# ruri-v3-310m prefix convention (applied by callers, not by get_embeddings)
DOCUMENT_PREFIX = "検索文書: "
QUERY_PREFIX = "検索クエリ: "

from langchain_community.embeddings import HuggingFaceEmbeddings
from settings import settings

_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings
