"""
ChromaDB client singleton and collection helpers.

Used by ingest.py (Phase 4A) and retriever.py (Phase 4C).
"""
import chromadb
from settings import settings

_client = None


def get_chroma_client() -> chromadb.HttpClient:
    global _client
    if _client is None:
        _client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
    return _client


def get_or_create_collection(datasource: str) -> chromadb.Collection:
    client = get_chroma_client()
    return client.get_or_create_collection(name=datasource)
