"""
OpenSearch client singleton, index mapping, and search pipeline helpers.

Index naming: {OS_INDEX_PREFIX}{datasource.lower()}
Mapping uses kuromoji analyzer for Japanese text and knn_vector for dense search.
"""
import logging
from settings import settings
from models.embedding_adapters import get_adapter

logger = logging.getLogger(__name__)

_os_client = None

_ANALYZERS = {
    "ja_text": {
        "type": "custom",
        "tokenizer": "kuromoji_tokenizer",
        "filter": [
            "icu_normalizer",
            "kuromoji_baseform",
            "kuromoji_part_of_speech",
            "ja_stop",
            "kuromoji_stemmer",
            "lowercase",
        ],
    },
    "ja_path": {
        "type": "custom",
        "tokenizer": "kuromoji_tokenizer",
        "filter": ["icu_normalizer", "lowercase"],
    },
}

_NORMALIZERS = {
    "lc_norm": {"type": "custom", "filter": ["lowercase"]},
}


def get_os_client():
    global _os_client
    if _os_client is None:
        from opensearchpy import OpenSearch
        _os_client = OpenSearch(settings.opensearch_url)
    return _os_client


def _index_name(datasource: str) -> str:
    return f"{settings.os_index_prefix}{datasource.lower()}"


def _build_mapping(dimension: int) -> dict:
    return {
        "settings": {
            "index": {"knn": True},
            "analysis": {
                "analyzer": _ANALYZERS,
                "normalizer": _NORMALIZERS,
            },
        },
        "mappings": {
            "properties": {
                "text": {"type": "text", "analyzer": "ja_text"},
                "file_name": {
                    "type": "text",
                    "analyzer": "ja_text",
                    "fields": {"keyword": {"type": "keyword", "normalizer": "lc_norm"}},
                },
                "title": {
                    "type": "text",
                    "analyzer": "ja_text",
                    "fields": {"keyword": {"type": "keyword", "normalizer": "lc_norm"}},
                },
                "section_path": {
                    "type": "text",
                    "analyzer": "ja_path",
                    "fields": {"keyword": {"type": "keyword"}},
                },
                "datasource": {"type": "keyword"},
                "source": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "ingested_at": {"type": "date"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                        "parameters": {"ef_construction": 256, "m": 16},
                    },
                },
            }
        },
    }


def _build_pipeline_body(pipeline_name: str) -> dict:
    return {
        "description": "Hybrid BM25 + kNN pipeline",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {"weights": [0.3, 0.7]},
                    },
                }
            }
        ],
    }


def get_or_create_index(datasource: str) -> str:
    """Idempotent index creation. Returns index name."""
    client = get_os_client()
    adapter = get_adapter(settings.embedding_model_name)
    index = _index_name(datasource)
    if not client.indices.exists(index=index):
        client.indices.create(index=index, body=_build_mapping(adapter.dimension))
        logger.info("Created OpenSearch index: %s (dim=%d)", index, adapter.dimension)
    return index


def get_or_create_search_pipeline() -> str:
    """Idempotent pipeline creation. Returns pipeline name."""
    client = get_os_client()
    pipeline_name = settings.hybrid_pipeline_name
    try:
        client.http.get(f"/_search/pipeline/{pipeline_name}")
        logger.debug("Search pipeline already exists: %s", pipeline_name)
    except Exception:
        client.http.put(
            f"/_search/pipeline/{pipeline_name}",
            body=_build_pipeline_body(pipeline_name),
        )
        logger.info("Created search pipeline: %s", pipeline_name)
    return pipeline_name
