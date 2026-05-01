"""
Batch ContextPrecision evaluation script using Langfuse Datasets.

Usage:
    uv run --directory app python evaluation/run_cp_eval.py \
        --dataset rag-cp-eval \
        --run-name bgem3-top10-2026-04-29 \
        --queries ../tests/data/cp-queries.yaml \
        --user nire

Requires Python 3.12 (pinned via .python-version). Python 3.13 is unverified;
3.14 has a known anyio/asyncio cleanup regression that masks OpenAI errors.
"""
import argparse
import asyncio
import hashlib
import logging
import os
import sys
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import _pyversion  # noqa: F401  -- triggers Python version guard on import

from evaluation.metrics import compute_context_precision
from rag.retriever import get_relevant_context
from settings import settings

logger = logging.getLogger(__name__)


def _get_langfuse():
    from langfuse import Langfuse  # lazy import — avoids pydantic.v1 load at collection time
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


def _load_queries(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("queries", [])


def _item_id(query: str, reference: str) -> str:
    """Deterministic 16-char hex ID for idempotent upsert."""
    return hashlib.sha256(f"{query}{reference}".encode()).hexdigest()[:16]


def _upsert_dataset_item(langfuse, dataset_name: str, item_data: dict):
    query = item_data["query"]
    reference = item_data.get("reference", "")
    return langfuse.create_dataset_item(
        dataset_name=dataset_name,
        id=_item_id(query, reference),
        input=query,
        expected_output=reference,
    )


async def _generate_answer(question: str, context: str) -> str:
    """Generate a minimal answer using the judge LLM for CP evaluation."""
    from models.provider import get_judge
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = get_judge()
    messages = [
        SystemMessage(content="Answer the question based on the context. Be concise."),
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {question}"),
    ]
    msg = await llm.ainvoke(messages)
    return msg.content


async def _evaluate_item(query: str, context: str, answer: str, reference: str) -> list:
    """Compute CP score; returns [Evaluation(...)] or [] when no score available."""
    from langfuse.experiment import Evaluation

    result = await compute_context_precision(query, context, answer, reference=reference)
    score = result.get("score")
    if score is None:
        logger.info("Skipping item (no reference or SKIP result): %s", query[:40])
        return []
    logger.info("Scored query '%s...': context_precision=%.4f", query[:40], score)
    return [Evaluation(name="context_precision", value=score)]


def run_eval(args) -> None:
    from models.provider import probe_endpoints
    asyncio.run(probe_endpoints())

    langfuse = _get_langfuse()
    try:
        queries = _load_queries(args.queries)
        langfuse.create_dataset(name=args.dataset)
        for q in queries:
            _upsert_dataset_item(langfuse, args.dataset, q)

        dataset_items = langfuse.get_dataset(args.dataset).items

        async def task(*, item, **kwargs):
            query = item.input
            context = await get_relevant_context(query, user=args.user)
            answer = await _generate_answer(query, context)
            return {"answer": answer, "context": context}

        async def evaluator(*, input, output, expected_output, metadata, **kwargs):
            if not output:
                return []
            return await _evaluate_item(
                input,
                output.get("context", ""),
                output.get("answer", ""),
                expected_output or "",
            )

        langfuse.run_experiment(
            name=args.dataset,
            run_name=args.run_name,
            data=dataset_items,
            task=task,
            evaluators=[evaluator],
        )
    finally:
        langfuse.flush()


def _parse_args():
    parser = argparse.ArgumentParser(description="Batch ContextPrecision evaluation via Langfuse Datasets")
    parser.add_argument("--dataset", required=True, help="Langfuse dataset name")
    parser.add_argument("--run-name", required=True, help="Run name for run-over-run comparison")
    parser.add_argument("--queries", required=True, help="Path to YAML queries file")
    parser.add_argument("--user", required=True, help="Username for ACL resolution")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
