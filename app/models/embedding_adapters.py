from dataclasses import dataclass


@dataclass
class EmbeddingAdapter:
    model_name: str
    document_prefix: str
    query_prefix: str
    normalize: bool
    dimension: int


_ADAPTERS: dict[str, EmbeddingAdapter] = {
    "cl-nagoya/ruri-v3-310m": EmbeddingAdapter(
        model_name="cl-nagoya/ruri-v3-310m",
        document_prefix="検索文書: ",
        query_prefix="検索クエリ: ",
        normalize=True,
        dimension=768,
    ),
    "BAAI/bge-m3": EmbeddingAdapter(
        model_name="BAAI/bge-m3",
        document_prefix="",
        query_prefix="",
        normalize=True,
        dimension=1024,
    ),
    "Qwen/Qwen3-Embedding-0.6B": EmbeddingAdapter(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        document_prefix="",
        query_prefix="Instruct: Retrieve semantically similar text.\nQuery: ",
        normalize=True,
        dimension=1024,
    ),
    "jinaai/jina-embeddings-v3": EmbeddingAdapter(
        model_name="jinaai/jina-embeddings-v3",
        document_prefix="",
        query_prefix="Represent the query for retrieving evidence documents: ",
        normalize=True,
        dimension=1024,
    ),
}


def get_adapter(model_name: str) -> EmbeddingAdapter:
    if not model_name:
        raise ValueError("model_name must not be empty")
    adapter = _ADAPTERS.get(model_name)
    if adapter is None:
        raise ValueError(
            f"Unknown embedding model: {model_name!r}. Known: {list(_ADAPTERS)}"
        )
    return adapter
