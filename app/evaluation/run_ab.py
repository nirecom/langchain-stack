"""
A/B evaluation script for embedding model comparison.

Ingests a datasource into per-model collections, then runs RAGAS metrics
(Response Relevancy, Faithfulness, Context Precision, optionally Context Recall)
against a shared query set to compare models.

Usage:
    uv run python app/evaluation/run_ab.py \\
        --datasource parents-docs \\
        --models ruri,qwen3,bgem3 \\
        --queries tests/data/ab-queries.yaml \\
        --output docs/embedding-ab-report.csv

Model shorthand → HuggingFace path:
    ruri    → cl-nagoya/ruri-v3-310m
    qwen3   → Qwen/Qwen3-Embedding-0.6B
    bgem3   → BAAI/bge-m3
    jinav3  → jinaai/jina-embeddings-v3
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import sys
import time
from pathlib import Path

import yaml

# Allow running as: uv run python app/evaluation/run_ab.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.embedding_adapters import get_adapter
from settings import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_ab")

MODEL_SHORTCUTS: dict[str, str] = {
    "ruri":   "cl-nagoya/ruri-v3-310m",
    "qwen3":  "Qwen/Qwen3-Embedding-0.6B",
    "bgem3":  "BAAI/bge-m3",
    "jinav3": "jinaai/jina-embeddings-v3",
}

SUPPORTED_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".txt", ".md", ".pptx", ".docx", ".doc"}


def _reset_embedding_singletons() -> None:
    """Reset cached embedding singletons so the next call loads the new model."""
    import models.embeddings as emb
    emb._query_embeddings = None
    emb._ingest_embeddings = None


def _collection_name(datasource: str, model_short: str) -> str:
    return f"{datasource}-{model_short}"


def _ingest_datasource_into_collection(src_datasource: str, collection_name: str) -> dict:
    """
    Ingest all files from /data/documents/{src_datasource}/ into {collection_name}.

    Returns dict with total_chunks, files_processed, errors.
    """
    from rag.ingest import ingest_file

    folder = Path("/data/documents") / src_datasource
    if not folder.is_dir():
        raise FileNotFoundError(f"Document folder not found: {folder}")

    total_chunks = 0
    files_processed = 0
    errors: list[dict] = []

    for file_path in sorted(folder.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            count = ingest_file(file_path, collection_name)
            total_chunks += count
            files_processed += 1
        except Exception as exc:
            logger.error("Failed to ingest %s: %s", file_path, exc)
            errors.append({"file": str(file_path), "error": str(exc)})

    return {"total_chunks": total_chunks, "files_processed": files_processed, "errors": errors}


async def _retrieve_from_collection(
    query: str,
    collection_name: str,
    model_name: str,
    n_results: int,
) -> str:
    """Retrieve context from a ChromaDB collection, bypassing ACL."""
    import chromadb.errors
    from models.chroma import get_chroma_client
    from models.embeddings import get_embeddings

    adapter = get_adapter(model_name)
    qvec = get_embeddings(role="query").embed_documents([adapter.query_prefix + query])[0]
    client = get_chroma_client()
    try:
        col = client.get_collection(name=collection_name)
    except chromadb.errors.NotFoundError:
        logger.error("Collection '%s' not found — was ingest skipped?", collection_name)
        return ""

    res = col.query(
        query_embeddings=[qvec],
        n_results=n_results,
        include=["documents", "distances"],
    )
    docs = (res.get("documents") or [[]])[0]
    return "\n\n---\n\n".join(doc.removeprefix(adapter.document_prefix) for doc in docs)


async def _retrieve_from_opensearch(
    query: str,
    datasource: str,
    model_name: str,
    search_mode: str,
    n_results: int,
) -> tuple[str, list[str]]:
    """Retrieve context from OpenSearch. Returns (context_text, source_filenames)."""
    from models.opensearch import get_os_client, _index_name
    from models.embeddings import get_embeddings
    from rag.retriever import _QUERY_BUILDERS, _format_hit

    adapter = get_adapter(model_name)
    qvec = get_embeddings(role="query").embed_documents([adapter.query_prefix + query])[0]
    builder = _QUERY_BUILDERS.get(search_mode, _QUERY_BUILDERS["hybrid+header"])
    body = builder(qvec, query, n_results)
    body["size"] = n_results
    body["_source"] = ["text", "file_name", "section_path", "source"]

    client = get_os_client()
    try:
        resp = client.search(
            index=_index_name(datasource),
            body=body,
            search_type="dfs_query_then_fetch",
        )
    except Exception as exc:
        logger.error("OpenSearch query failed: %s", exc)
        return "", []

    hits = (resp.get("hits") or {}).get("hits") or []
    sources = [h["_source"].get("source", "") for h in hits]
    context = "\n\n---\n\n".join(_format_hit(h["_source"]) for h in hits)
    return context, sources


async def _generate_answer(question: str, context: str) -> str:
    from langchain_core.messages import HumanMessage
    from models.provider import get_reasoner

    llm = get_reasoner()
    prompt = (
        f"Answer the following question based only on the provided context.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content


async def _evaluate_query(
    query_item: dict,
    collection_name: str,
    model_name: str,
    n_results: int,
    *,
    backend: str = "opensearch",
    search_mode: str = "hybrid+header",
    datasource: str = "",
) -> dict:
    """Run retrieval + answer generation + RAGAS metrics for a single query."""
    from evaluation.metrics import (
        compute_context_precision,
        compute_faithfulness,
        compute_response_relevancy,
    )

    question = query_item["query"]
    lang = query_item.get("language", "unknown")
    expected = query_item.get("expected_answer", "")
    expected_source = query_item.get("expected_source", "")

    t0 = time.monotonic()
    if backend == "opensearch":
        context, retrieved_sources = await _retrieve_from_opensearch(
            question, datasource, model_name, search_mode, n_results,
        )
    else:
        context = await _retrieve_from_collection(question, collection_name, model_name, n_results)
        retrieved_sources = []
    retrieve_ms = (time.monotonic() - t0) * 1000

    source_hit: int | None = None
    if backend == "opensearch" and expected_source:
        source_hit = 1 if any(expected_source in s for s in retrieved_sources) else 0

    if not context:
        return {
            "query": question,
            "language": lang,
            "context_found": False,
            "retrieve_ms": round(retrieve_ms),
            "search_mode": search_mode,
            "backend": backend,
            "relevancy": None,
            "faithfulness": None,
            "context_precision": None,
            "context_recall": None,
            "source_hit_at_k": source_hit,
            "answer": "",
        }

    answer = await _generate_answer(question, context)

    rel = await compute_response_relevancy(question, answer)
    faith = await compute_faithfulness(question, context, answer)
    cp = await compute_context_precision(question, context, answer, reference=expected)

    cr: float | None = None
    if expected:
        try:
            from ragas import SingleTurnSample
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import ContextRecall
            from models.provider import get_judge

            metric = ContextRecall(llm=LangchainLLMWrapper(get_judge()))
            sample = SingleTurnSample(
                user_input=question,
                retrieved_contexts=[context],
                response=answer,
                reference=expected,
            )
            cr_score = await metric.single_turn_ascore(sample)
            cr = round(float(cr_score), 4)
        except Exception as exc:
            logger.error("RAGAS ContextRecall evaluation failed: %s", exc)

    return {
        "query": question,
        "language": lang,
        "context_found": True,
        "retrieve_ms": round(retrieve_ms),
        "search_mode": search_mode,
        "backend": backend,
        "relevancy": rel["score"],
        "faithfulness": faith["score"],
        "context_precision": cp["score"],
        "context_recall": cr,
        "source_hit_at_k": source_hit,
        "answer": answer,
    }


def _load_queries(path: str, datasource_filter: str | None) -> list[dict]:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    items: list[dict] = raw.get("queries", raw) if isinstance(raw, dict) else raw
    if datasource_filter:
        items = [q for q in items if q.get("datasource", datasource_filter) == datasource_filter]
    return items


def _write_csv(rows: list[dict], output_path: str) -> None:
    if not rows:
        logger.warning("No rows to write")
        return
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "datasource", "collection", "query", "language",
              "context_found", "retrieve_ms", "search_mode", "backend",
              "relevancy", "faithfulness", "context_precision", "context_recall",
              "source_hit_at_k"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Results written to %s", out)


async def run_ab(
    datasource: str,
    model_shorts: list[str],
    queries_path: str,
    output_path: str,
    n_results: int,
    skip_ingest: bool,
    search_modes: list[str] | None = None,
    backend: str = "opensearch",
    header_source: str = "loader",
) -> None:
    queries = _load_queries(queries_path, datasource)
    if not queries:
        logger.error("No queries found for datasource '%s' in %s", datasource, queries_path)
        return

    logger.info("Loaded %d queries for datasource '%s'", len(queries), datasource)
    effective_modes = search_modes or (["hybrid+header"] if backend == "opensearch" else [None])
    all_rows: list[dict] = []

    for short in model_shorts:
        hf_name = MODEL_SHORTCUTS[short]
        collection = _collection_name(datasource, short)

        logger.info("=== Model: %s (%s) backend=%s ===", short, hf_name, backend)

        settings.embedding_model_name = hf_name
        _reset_embedding_singletons()

        if not skip_ingest:
            target = datasource if backend == "opensearch" else collection
            logger.info("Ingesting '%s' → '%s'", datasource, target)
            result = _ingest_datasource_into_collection(datasource, target)
            logger.info(
                "Ingest complete: %d chunks from %d files (%d errors)",
                result["total_chunks"], result["files_processed"], len(result["errors"]),
            )
            if result["errors"]:
                for err in result["errors"]:
                    logger.warning("  Ingest error: %s — %s", err["file"], err["error"])
        else:
            logger.info("Skipping ingest (--skip-ingest)")

        for i, query_item in enumerate(queries, 1):
            for mode in effective_modes:
                logger.info(
                    "  [%d/%d] mode=%s %s query: %s",
                    i, len(queries), mode or "chroma",
                    query_item.get("language", "?"), query_item["query"][:60],
                )
                row = await _evaluate_query(
                    query_item, collection, hf_name, n_results,
                    backend=backend,
                    search_mode=mode or "hybrid+header",
                    datasource=datasource,
                )
                row["model"] = short
                row["datasource"] = datasource
                row["collection"] = collection
                all_rows.append(row)

        logger.info(
            "Model '%s' done: relevancy=%.3f faith=%.3f cp=%.3f",
            short,
            _mean([r["relevancy"] for r in all_rows if r["model"] == short and r["relevancy"] is not None]),
            _mean([r["faithfulness"] for r in all_rows if r["model"] == short and r["faithfulness"] is not None]),
            _mean([r["context_precision"] for r in all_rows if r["model"] == short and r["context_precision"] is not None]),
        )

    _write_csv(all_rows, output_path)
    _print_summary(all_rows, model_shorts)


def _mean(values: list) -> float:
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else 0.0


def _print_summary(rows: list[dict], model_shorts: list[str]) -> None:
    print("\n=== A/B Evaluation Summary ===")
    header = f"{'Model':<10} {'Lang':<5} {'Relevancy':>10} {'Faithful':>10} {'CtxPrec':>10} {'CtxRecall':>10} {'p50ms':>7}"
    print(header)
    print("-" * len(header))

    for short in model_shorts:
        for lang in ("en", "ja"):
            subset = [r for r in rows if r["model"] == short and r["language"] == lang]
            if not subset:
                continue
            rel = _mean([r["relevancy"] for r in subset])
            faith = _mean([r["faithfulness"] for r in subset])
            cp = _mean([r["context_precision"] for r in subset])
            cr = _mean([r["context_recall"] for r in subset])
            ms_vals = sorted(r["retrieve_ms"] for r in subset)
            p50 = ms_vals[len(ms_vals) // 2] if ms_vals else 0
            print(
                f"{short:<10} {lang:<5} {rel:>10.3f} {faith:>10.3f} {cp:>10.3f} "
                f"{cr if cr else '':>10} {p50:>7}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A/B evaluation: compare embedding models via RAGAS metrics"
    )
    parser.add_argument(
        "--datasource", required=True,
        help="Datasource name (maps to /data/documents/<name>/ and ACL config)",
    )
    parser.add_argument(
        "--models", required=True,
        help=f"Comma-separated model shorthand list. Known: {', '.join(MODEL_SHORTCUTS)}",
    )
    parser.add_argument(
        "--queries", default="tests/data/ab-queries.yaml",
        help="Path to queries YAML file (default: tests/data/ab-queries.yaml)",
    )
    parser.add_argument(
        "--output", default="docs/embedding-ab-report.csv",
        help="Output CSV path (default: docs/embedding-ab-report.csv)",
    )
    parser.add_argument(
        "--n-results", type=int, default=3,
        help="Number of chunks to retrieve per query (default: 3)",
    )
    parser.add_argument(
        "--skip-ingest", action="store_true",
        help="Skip ingest step; use existing collections (for re-running evaluation only)",
    )
    parser.add_argument(
        "--search-modes", default=None,
        help=(
            "Comma-separated search modes for OpenSearch backend. "
            "e.g.: dense,header+dense,hybrid,hybrid+header (default: hybrid+header)"
        ),
    )
    parser.add_argument(
        "--backend", default="opensearch", choices=["opensearch", "chroma"],
        help="Vector backend to use (default: opensearch)",
    )
    parser.add_argument(
        "--header-source", default="loader", choices=["loader", "llm", "none"],
        help="PDF title extraction strategy (default: loader)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    shorts = [s.strip() for s in args.models.split(",")]
    unknown = [s for s in shorts if s not in MODEL_SHORTCUTS]
    if unknown:
        print(f"ERROR: Unknown model shorthand(s): {unknown}. Known: {list(MODEL_SHORTCUTS)}")
        sys.exit(1)

    search_modes = (
        [m.strip() for m in args.search_modes.split(",")]
        if args.search_modes
        else None
    )

    asyncio.run(
        run_ab(
            datasource=args.datasource,
            model_shorts=shorts,
            queries_path=args.queries,
            output_path=args.output,
            n_results=args.n_results,
            skip_ingest=args.skip_ingest,
            search_modes=search_modes,
            backend=args.backend,
            header_source=args.header_source,
        )
    )
