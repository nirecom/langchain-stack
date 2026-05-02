"""
OpenSearch retriever with ACL-scoped multi-index search and 4 query modes.

Modes (controlled by settings.search_mode):
  dense         - kNN only
  header+dense  - kNN + BM25 boost on title/file_name fields
  hybrid        - BM25 (text) + kNN via search pipeline
  hybrid+header - BM25 (text+title+file_name+section_path boosted) + kNN via pipeline

ACL is enforced at the index level: each datasource maps to a separate index,
and only permitted indices are queried.
"""
import logging
import re

from models.opensearch import get_os_client, _index_name
from models.embeddings import get_embeddings
from models.embedding_adapters import get_adapter
from rag.access_control import get_permitted_datasources_for_user
from rag.audit import log_retrieve_event
from settings import settings

logger = logging.getLogger(__name__)

_MIN_KNN_K = 10  # prevents min_max normalizer from zeroing single-result queries

_COUNT_PATTERNS = re.compile(r"何件|何個|いくつ|何本|何冊|何例|件数|何事例|何ケース")


def _is_counting_query(query: str) -> bool:
    return bool(_COUNT_PATTERNS.search(query))


def _build_title_bm25(query_vector: list, query_text: str, k: int) -> dict:
    return {
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": ["title^3.0", "file_name^3.0", "section_path^1.5"],
                "type": "best_fields",
            }
        }
    }


def _current_adapter():
    name = settings.embedding_model_name
    if not isinstance(name, str):
        name = "cl-nagoya/ruri-v3-310m"
    return get_adapter(name)


def _build_dense(query_vector: list, query_text: str, k: int) -> dict:
    effective_k = max(k, _MIN_KNN_K)
    return {"query": {"knn": {"embedding": {"vector": query_vector, "k": effective_k}}}}


def _build_header_dense(query_vector: list, query_text: str, k: int) -> dict:
    effective_k = max(k, _MIN_KNN_K)
    return {
        "query": {
            "bool": {
                "must": [{"knn": {"embedding": {"vector": query_vector, "k": effective_k}}}],
                "should": [
                    {"match": {"title": {"query": query_text, "boost": 2.5}}},
                    {"match": {"file_name": {"query": query_text, "boost": 2.0}}},
                ],
            }
        }
    }


def _build_hybrid(query_vector: list, query_text: str, k: int) -> dict:
    effective_k = max(k, _MIN_KNN_K)
    return {
        "query": {
            "hybrid": {
                "queries": [
                    {"multi_match": {"query": query_text, "fields": ["text"]}},
                    {"knn": {"embedding": {"vector": query_vector, "k": effective_k}}},
                ]
            }
        },
        "search_pipeline": settings.hybrid_pipeline_name,
    }


def _build_hybrid_header(query_vector: list, query_text: str, k: int) -> dict:
    effective_k = max(k, _MIN_KNN_K)
    return {
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": [
                                "text^1.0",
                                "title^2.5",
                                "file_name^2.0",
                                "section_path^1.5",
                            ],
                        }
                    },
                    {"knn": {"embedding": {"vector": query_vector, "k": effective_k}}},
                ]
            }
        },
        "search_pipeline": settings.hybrid_pipeline_name,
    }


_QUERY_BUILDERS = {
    "dense": _build_dense,
    "header+dense": _build_header_dense,
    "hybrid": _build_hybrid,
    "hybrid+header": _build_hybrid_header,
}


def _format_hit(src: dict) -> str:
    file_name = src.get("file_name", "")
    section = src.get("section_path") or ""
    text = src.get("text", "")
    label = f"{file_name} | {section}" if section else file_name
    return f"[Source: {label}]\n{text}" if label else text


async def get_relevant_context(
    query: str,
    *,
    user: str,
    n_results: int | None = None,
    search_mode: str | None = None,
    datasources: list[str] | None = None,
) -> str:
    """
    Retrieve RAG context for *query*, restricted to ACL-permitted indices for *user*.

    Args:
        datasources: Optional list to narrow search to specific datasources.
            Intersected with user's ACL — cannot grant access beyond what user is permitted.
    Returns "" on failure or when no results are found.
    """
    if not query.strip():
        return ""

    k = n_results if n_results is not None else settings.rag_top_k
    mode = search_mode or settings.search_mode
    adapter = _current_adapter()

    if _is_counting_query(query):
        builder = _build_title_bm25
        k = max(k, 30)
    else:
        builder = _QUERY_BUILDERS.get(mode, _build_hybrid_header)

    permitted = set(get_permitted_datasources_for_user(user))
    if datasources is not None:
        permitted = permitted & set(datasources)
    if not permitted:
        logger.warning("RAG: no datasources permitted for user '%s'", user)
        log_retrieve_event(
            user=user, datasources_queried=[], query=query, hits=0,
            status="no_permitted_datasources",
        )
        return ""

    qvec = get_embeddings(role="query").embed_documents([adapter.query_prefix + query])[0]

    indices = [_index_name(ds) for ds in sorted(permitted)]
    body = builder(qvec, query, k)
    body["size"] = k
    body["_source"] = ["text", "file_name", "section_path", "source"]

    client = get_os_client()
    try:
        resp = client.search(
            index=",".join(indices),
            body=body,
            search_type="dfs_query_then_fetch",
            params={"ignore_unavailable": "true"},
        )
    except Exception as e:
        logger.error("OpenSearch query failed: %s", e)
        log_retrieve_event(
            user=user, datasources_queried=sorted(permitted),
            query=query, hits=0, status="error", error=str(e),
        )
        return ""

    hits = (resp.get("hits") or {}).get("hits") or []
    top = hits[:k]

    log_retrieve_event(
        user=user, datasources_queried=sorted(permitted),
        query=query, hits=len(top),
    )

    return "\n\n---\n\n".join(_format_hit(h["_source"]) for h in top) if top else ""
