"""
Document ingestion pipeline.

Loads files (8 formats), splits into chunks, embeds with ruri-v3-310m,
and stores in ChromaDB with per-datasource collection isolation.
"""
import logging
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from models.embeddings import get_embeddings
from models.embedding_adapters import get_adapter
from models.opensearch import get_os_client, get_or_create_index, get_or_create_search_pipeline, _index_name
from rag.headers import extract_headers
from settings import settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".txt", ".md", ".pptx", ".docx", ".doc"}

_LOADER_TYPE_MAP = {
    ".pdf": "pdf", ".md": "md", ".docx": "docx", ".pptx": "pptx",
    ".xlsx": "xlsx", ".xls": "xlsx", ".txt": "txt", ".doc": "doc",
}


def _current_adapter():
    name = settings.embedding_model_name
    if not isinstance(name, str):
        name = "cl-nagoya/ruri-v3-310m"
    return get_adapter(name)


def _load_documents(file_path: Path):
    """Load documents using format-specific loader."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        from langchain_community.document_loaders import PyMuPDFLoader
        return PyMuPDFLoader(str(file_path)).load()

    if suffix in (".xlsx", ".xls"):
        from langchain_community.document_loaders import UnstructuredExcelLoader
        return UnstructuredExcelLoader(str(file_path), mode="elements").load()

    if suffix == ".txt":
        from langchain_community.document_loaders import TextLoader
        return TextLoader(str(file_path)).load()

    if suffix == ".md":
        from langchain_community.document_loaders import UnstructuredMarkdownLoader
        return UnstructuredMarkdownLoader(str(file_path), mode="elements").load()

    if suffix == ".pptx":
        from langchain_community.document_loaders import UnstructuredPowerPointLoader
        return UnstructuredPowerPointLoader(str(file_path), mode="elements").load()

    if suffix == ".docx":
        from langchain_community.document_loaders import UnstructuredWordDocumentLoader
        return UnstructuredWordDocumentLoader(str(file_path), mode="elements").load()

    if suffix == ".doc":
        return _load_doc_with_antiword(file_path)

    raise ValueError(f"Unsupported file extension: {suffix}")


def _load_doc_with_antiword(file_path: Path):
    """Convert legacy .doc to text using antiword, then load as TextLoader."""
    from langchain_community.document_loaders import TextLoader

    result = subprocess.run(
        ["antiword", str(file_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"antiword failed: {result.stderr}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp.write(result.stdout)
        tmp_path = tmp.name

    try:
        return TextLoader(tmp_path).load()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _load_with_headers(file_path: Path) -> tuple[list, str, list]:
    """Load documents and extract title + per-doc section_path."""
    documents = _load_documents(file_path)
    loader_type = _LOADER_TYPE_MAP.get(file_path.suffix.lower(), "txt")
    title, section_paths = extract_headers(file_path, documents, loader_type)
    sp_list = section_paths if section_paths else [None] * len(documents)
    for doc, sp in zip(documents, sp_list):
        doc.metadata["section_path"] = sp
    return documents, title, sp_list


def _split_documents(documents):
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.ingest_chunk_size,
        chunk_overlap=settings.ingest_chunk_overlap,
    )
    return splitter.split_documents(documents)


def ingest_file(file_path: Path, datasource: str, *, original_filename: str | None = None) -> int:
    """
    Ingest a single file into ChromaDB.

    Args:
        file_path: Path to the file to ingest
        datasource: Target datasource (ChromaDB collection name)
        original_filename: Original filename for metadata (uses file_path.name if omitted)

    Returns:
        Number of chunks stored
    """
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {file_path.suffix}")

    source_name = original_filename or file_path.name
    logger.info("Ingesting %s into datasource '%s'", source_name, datasource)

    documents, file_title, _ = _load_with_headers(file_path)
    if not documents:
        logger.warning("No content extracted from %s", source_name)
        return 0

    chunks = _split_documents(documents)
    if not chunks:
        logger.warning("No chunks after splitting %s", source_name)
        return 0

    index = get_or_create_index(datasource)
    get_or_create_search_pipeline()
    client = get_os_client()
    embeddings = get_embeddings(role="ingest")
    adapter = _current_adapter()
    now = datetime.now(timezone.utc).isoformat()

    # Delete existing chunks for this file (duplicate ingestion handling)
    client.delete_by_query(
        index=index,
        body={"query": {"term": {"source": source_name}}},
    )

    # Embed with document_prefix (for bge-m3 / ruri); store raw text for BM25
    embed_texts = [adapter.document_prefix + chunk.page_content for chunk in chunks]
    vectors = embeddings.embed_documents(embed_texts)

    from opensearchpy import helpers as os_helpers
    actions = [
        {
            "_index": index,
            "_id": f"{source_name}::{i}",
            "_source": {
                "text": chunk.page_content,
                "embedding": vectors[i],
                "file_name": source_name,
                "title": file_title,
                "section_path": chunk.metadata.get("section_path"),
                "datasource": datasource,
                "source": source_name,
                "chunk_index": i,
                "ingested_at": now,
            },
        }
        for i, chunk in enumerate(chunks)
    ]
    os_helpers.bulk(client, actions)

    logger.info("Stored %d chunks from %s in '%s'", len(chunks), source_name, datasource)
    return len(chunks)


def ingest_folder(datasource: str) -> dict:
    """
    Ingest all supported files in data/documents/{datasource}/.

    Returns:
        dict with keys: total_chunks, files_processed, errors
    """
    folder = Path("/data/documents") / datasource
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder}")

    total_chunks = 0
    files_processed = 0
    errors = []

    for file_path in sorted(folder.rglob("*")):
        if not file_path.is_file():
            continue
        if any(p.name.startswith("_") for p in file_path.relative_to(folder).parents):
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            relative = file_path.relative_to(folder)
            count = ingest_file(file_path, datasource, original_filename=str(relative))
            total_chunks += count
            files_processed += 1
        except Exception as e:
            logger.error("Failed to ingest %s: %s", file_path, e)
            errors.append({"file": str(file_path), "error": str(e)})

    return {
        "total_chunks": total_chunks,
        "files_processed": files_processed,
        "errors": errors,
    }


def list_files(datasource: str) -> list[dict]:
    """List unique filenames and their chunk counts in a datasource."""
    client = get_os_client()
    index = _index_name(datasource)
    resp = client.search(
        index=index,
        body={"size": 0, "aggs": {"files": {"terms": {"field": "source", "size": 10000}}}},
    )
    buckets = resp["aggregations"]["files"]["buckets"]
    return [
        {"filename": b["key"], "chunk_count": b["doc_count"]}
        for b in sorted(buckets, key=lambda b: b["key"])
    ]


def delete_file(datasource: str, filename: str) -> int:
    """Delete all chunks for a specific file from a datasource. Returns deleted count."""
    client = get_os_client()
    index = _index_name(datasource)
    resp = client.delete_by_query(
        index=index,
        body={"query": {"term": {"source": filename}}},
    )
    deleted = resp.get("deleted", 0)
    if deleted == 0:
        raise ValueError(f"No chunks found for '{filename}' in '{datasource}'")
    logger.info("Deleted %d chunks for '%s' from '%s'", deleted, filename, datasource)
    return deleted


def dry_run_file(file_path: Path, *, original_filename: str | None = None) -> dict:
    """Preview chunking without writing to ChromaDB."""
    source_name = original_filename or file_path.name
    documents = _load_documents(file_path)
    if not documents:
        return {"filename": source_name, "total_chunks": 0, "chunks": []}
    chunks = _split_documents(documents)
    return {
        "filename": source_name,
        "total_chunks": len(chunks),
        "chunks": [
            {
                "chunk_index": i,
                "char_count": len(chunk.page_content),
                "preview": chunk.page_content[:200],
            }
            for i, chunk in enumerate(chunks)
        ],
    }


def delete_collection(datasource: str) -> bool:
    """Delete an OpenSearch index for the datasource."""
    client = get_os_client()
    index = _index_name(datasource)
    try:
        client.indices.delete(index=index)
        logger.info("Deleted index '%s'", index)
        return True
    except Exception as e:
        logger.error("Failed to delete index '%s': %s", index, e)
        raise
