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
from models.chroma import get_or_create_collection
from settings import settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".txt", ".md", ".pptx", ".docx", ".doc"}
DOCUMENT_PREFIX = "検索文書: "


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

    documents = _load_documents(file_path)
    if not documents:
        logger.warning("No content extracted from %s", source_name)
        return 0

    chunks = _split_documents(documents)
    if not chunks:
        logger.warning("No chunks after splitting %s", source_name)
        return 0

    collection = get_or_create_collection(datasource)
    embeddings = get_embeddings()
    now = datetime.now(timezone.utc).isoformat()

    # Delete existing chunks for this file (duplicate ingestion handling)
    existing = collection.get(where={"source": source_name})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        logger.info("Deleted %d existing chunks for %s", len(existing["ids"]), source_name)

    # Prepare texts with prefix and metadata
    texts = [DOCUMENT_PREFIX + chunk.page_content for chunk in chunks]
    vectors = embeddings.embed_documents(texts)
    ids = [f"{source_name}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": source_name,
            "datasource": datasource,
            "chunk_index": i,
            "ingested_at": now,
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
    )

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
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            count = ingest_file(file_path, datasource)
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
    from collections import Counter
    from models.chroma import get_chroma_client

    client = get_chroma_client()
    collection = client.get_collection(name=datasource)
    result = collection.get(include=["metadatas"])
    counts = Counter(m["source"] for m in result["metadatas"])
    return [{"filename": name, "chunk_count": cnt} for name, cnt in sorted(counts.items())]


def delete_file(datasource: str, filename: str) -> int:
    """Delete all chunks for a specific file from a datasource. Returns deleted count."""
    from models.chroma import get_chroma_client

    client = get_chroma_client()
    collection = client.get_collection(name=datasource)
    existing = collection.get(where={"source": filename})
    if not existing["ids"]:
        raise ValueError(f"No chunks found for '{filename}' in '{datasource}'")
    collection.delete(ids=existing["ids"])
    logger.info("Deleted %d chunks for '%s' from '%s'", len(existing["ids"]), filename, datasource)
    return len(existing["ids"])


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
    """Delete a ChromaDB collection."""
    from models.chroma import get_chroma_client

    client = get_chroma_client()
    try:
        client.delete_collection(name=datasource)
        logger.info("Deleted collection '%s'", datasource)
        return True
    except Exception as e:
        logger.error("Failed to delete collection '%s': %s", datasource, e)
        raise
