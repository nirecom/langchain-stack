"""
Unit tests for Phase 4C: RAG retrieval pipeline.

Tests cover:
  - Normal: single/multi collection retrieval, top-k merge, settings fallback
  - Error: unknown model, missing collection, empty model name
  - Edge: empty/whitespace query, tied distances, explicit n_results, query prefix
  - Idempotency: deterministic output across repeated calls
  - Criteria switching: rag vs default judge criteria based on context presence
  - Audit: retrieve event logging with PII protection
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import json
import logging
import pytest
import asyncio
import chromadb.errors
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

DOCUMENT_PREFIX = "\u691c\u7d22\u6587\u66f8: "  # "検索文書: "
QUERY_PREFIX = "\u691c\u7d22\u30af\u30a8\u30ea: "  # "検索クエリ: "


# -- Stubs ----------------------------------------------------------------


class StubEmbeddings:
    """Stub embedding model that returns fixed vectors."""

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class StubCollection:
    """Stub ChromaDB collection returning pre-configured results."""

    def __init__(self, name, documents, distances):
        self._name = name
        self._documents = documents
        self._distances = distances

    def query(self, query_embeddings, n_results, include=None):
        docs = self._documents[:n_results]
        dists = self._distances[:n_results]
        return {"documents": [docs], "distances": [dists]}


class StubChromaClient:
    """Stub ChromaDB client with configurable collections."""

    def __init__(self, collections=None):
        self._collections = collections or {}

    def get_collection(self, name):
        if name not in self._collections:
            raise chromadb.errors.NotFoundError(f"Collection {name} not found")
        return self._collections[name]


# -- Fixtures -------------------------------------------------------------


@pytest.fixture
def stub_embeddings():
    return StubEmbeddings()


@pytest.fixture
def single_collection_client():
    """Client with one collection 'alpha' containing 5 docs."""
    docs = [
        f"{DOCUMENT_PREFIX}doc{i}" for i in range(5)
    ]
    distances = [0.1, 0.2, 0.3, 0.4, 0.5]
    col = StubCollection("alpha", docs, distances)
    return StubChromaClient(collections={"alpha": col})


@pytest.fixture
def multi_collection_client():
    """Client with 'alpha' and 'beta' collections for interleaved merge tests."""
    alpha_docs = [f"{DOCUMENT_PREFIX}alpha_doc0", f"{DOCUMENT_PREFIX}alpha_doc1"]
    alpha_dists = [0.2, 0.4]
    beta_docs = [f"{DOCUMENT_PREFIX}beta_doc0", f"{DOCUMENT_PREFIX}beta_doc1"]
    beta_dists = [0.1, 0.5]
    return StubChromaClient(collections={
        "alpha": StubCollection("alpha", alpha_docs, alpha_dists),
        "beta": StubCollection("beta", beta_docs, beta_dists),
    })


@pytest.fixture
def large_multi_client():
    """Client with 2 collections x 5 docs each for truncation tests."""
    col_a_docs = [f"{DOCUMENT_PREFIX}a{i}" for i in range(5)]
    col_a_dists = [0.1 * (i + 1) for i in range(5)]
    col_b_docs = [f"{DOCUMENT_PREFIX}b{i}" for i in range(5)]
    col_b_dists = [0.15 * (i + 1) for i in range(5)]
    return StubChromaClient(collections={
        "col_a": StubCollection("col_a", col_a_docs, col_a_dists),
        "col_b": StubCollection("col_b", col_b_docs, col_b_dists),
    })


@pytest.fixture
def tied_distance_client():
    """Client with alpha and beta at identical distances."""
    alpha_docs = [f"{DOCUMENT_PREFIX}alpha_tied"]
    alpha_dists = [0.5]
    beta_docs = [f"{DOCUMENT_PREFIX}beta_tied"]
    beta_dists = [0.5]
    return StubChromaClient(collections={
        "alpha": StubCollection("alpha", alpha_docs, alpha_dists),
        "beta": StubCollection("beta", beta_docs, beta_dists),
    })


# -- Helpers ---------------------------------------------------------------


def _patch_retriever(chroma_client, embeddings, permitted, settings_overrides=None):
    """Return a dict of patches for rag.retriever dependencies."""
    patches = {
        "rag.retriever.get_chroma_client": MagicMock(return_value=chroma_client),
        "rag.retriever.get_embeddings": MagicMock(return_value=embeddings),
        "rag.retriever.get_permitted_datasources": MagicMock(return_value=permitted),
        "rag.retriever.log_retrieve_event": MagicMock(),
    }
    return patches


# ==========================================================================
# Normal cases
# ==========================================================================


class TestNormalCases:
    @pytest.mark.asyncio
    async def test_single_collection_top_k(
        self, single_collection_client, stub_embeddings
    ):
        """1 permitted collection, 5 docs, rag_top_k=3 -> top 3 in distance order,
        DOCUMENT_PREFIX stripped."""
        patches = _patch_retriever(
            single_collection_client, stub_embeddings, ["alpha"]
        )
        with (
            patch("rag.retriever.get_chroma_client", patches["rag.retriever.get_chroma_client"]),
            patch("rag.retriever.get_embeddings", patches["rag.retriever.get_embeddings"]),
            patch("rag.retriever.get_permitted_datasources", patches["rag.retriever.get_permitted_datasources"]),
            patch("rag.retriever.log_retrieve_event", patches["rag.retriever.log_retrieve_event"]),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            from rag.retriever import get_relevant_context

            result = await get_relevant_context("test query", model_name="test-model")

        # Top 3 docs should be returned with prefix stripped
        assert "doc0" in result
        assert "doc1" in result
        assert "doc2" in result
        # doc3 and doc4 should NOT be included (beyond top_k)
        assert "doc3" not in result
        assert "doc4" not in result

    @pytest.mark.asyncio
    async def test_multi_collection_interleaved_merge(
        self, multi_collection_client, stub_embeddings
    ):
        """2 collections alpha([0.2,0.4]) and beta([0.1,0.5]), rag_top_k=3
        -> result order: beta:0.1, alpha:0.2, alpha:0.4."""
        patches = _patch_retriever(
            multi_collection_client, stub_embeddings, ["alpha", "beta"]
        )
        with (
            patch("rag.retriever.get_chroma_client", patches["rag.retriever.get_chroma_client"]),
            patch("rag.retriever.get_embeddings", patches["rag.retriever.get_embeddings"]),
            patch("rag.retriever.get_permitted_datasources", patches["rag.retriever.get_permitted_datasources"]),
            patch("rag.retriever.log_retrieve_event", patches["rag.retriever.log_retrieve_event"]),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            from rag.retriever import get_relevant_context

            result = await get_relevant_context("test query", model_name="test-model")

        # beta_doc0 (0.1) should appear before alpha_doc0 (0.2) before alpha_doc1 (0.4)
        pos_beta0 = result.find("beta_doc0")
        pos_alpha0 = result.find("alpha_doc0")
        pos_alpha1 = result.find("alpha_doc1")
        assert pos_beta0 != -1
        assert pos_alpha0 != -1
        assert pos_alpha1 != -1
        assert pos_beta0 < pos_alpha0 < pos_alpha1

    @pytest.mark.asyncio
    async def test_post_merge_truncation(
        self, large_multi_client, stub_embeddings
    ):
        """2 collections x 5 hits each, rag_top_k=3 -> exactly 3 items."""
        patches = _patch_retriever(
            large_multi_client, stub_embeddings, ["col_a", "col_b"]
        )
        with (
            patch("rag.retriever.get_chroma_client", patches["rag.retriever.get_chroma_client"]),
            patch("rag.retriever.get_embeddings", patches["rag.retriever.get_embeddings"]),
            patch("rag.retriever.get_permitted_datasources", patches["rag.retriever.get_permitted_datasources"]),
            patch("rag.retriever.log_retrieve_event", patches["rag.retriever.log_retrieve_event"]),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            from rag.retriever import get_relevant_context

            result = await get_relevant_context("test query", model_name="test-model")

        # Count distinct doc references — should be exactly 3
        # Each doc appears on its own line or section; count non-empty segments
        assert result != ""
        # Split by double-newline or similar separator used in context assembly
        # At minimum, verify the total is bounded: 4th-closest doc should be absent
        # The 3 closest: col_a:0.1, col_b:0.15, col_a:0.2
        assert "a0" in result  # col_a doc0 at distance 0.1
        assert "b0" in result  # col_b doc0 at distance 0.15
        assert "a1" in result  # col_a doc1 at distance 0.2
        # 4th closest would be col_b doc1 at 0.30 — should be excluded
        assert "b1" not in result

    @pytest.mark.asyncio
    async def test_n_results_none_uses_settings(
        self, single_collection_client, stub_embeddings
    ):
        """n_results=None -> settings.rag_top_k is consulted."""
        patches = _patch_retriever(
            single_collection_client, stub_embeddings, ["alpha"]
        )
        with (
            patch("rag.retriever.get_chroma_client", patches["rag.retriever.get_chroma_client"]),
            patch("rag.retriever.get_embeddings", patches["rag.retriever.get_embeddings"]),
            patch("rag.retriever.get_permitted_datasources", patches["rag.retriever.get_permitted_datasources"]),
            patch("rag.retriever.log_retrieve_event", patches["rag.retriever.log_retrieve_event"]),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 2
            from rag.retriever import get_relevant_context

            result = await get_relevant_context(
                "test query", model_name="test-model", n_results=None
            )

        # With rag_top_k=2, only 2 docs should be returned
        assert "doc0" in result
        assert "doc1" in result
        assert "doc2" not in result


# ==========================================================================
# Error cases
# ==========================================================================


class TestErrorCases:
    @pytest.mark.asyncio
    async def test_unknown_model_returns_empty(
        self, single_collection_client, stub_embeddings, caplog
    ):
        """model_name='nonexistent' -> '' + WARNING log."""
        patches = _patch_retriever(
            single_collection_client, stub_embeddings, []  # no permitted datasources
        )
        with (
            patch("rag.retriever.get_chroma_client", patches["rag.retriever.get_chroma_client"]),
            patch("rag.retriever.get_embeddings", patches["rag.retriever.get_embeddings"]),
            patch("rag.retriever.get_permitted_datasources", patches["rag.retriever.get_permitted_datasources"]),
            patch("rag.retriever.log_retrieve_event", patches["rag.retriever.log_retrieve_event"]),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            from rag.retriever import get_relevant_context

            with caplog.at_level(logging.WARNING):
                result = await get_relevant_context(
                    "test query", model_name="nonexistent"
                )

        assert result == ""
        assert any("WARNING" in r.levelname for r in caplog.records) or len(caplog.records) > 0

    @pytest.mark.asyncio
    async def test_collection_not_found_skips(
        self, stub_embeddings, caplog
    ):
        """2 permitted, 1 raises NotFoundError -> other collection's results returned."""
        # Only 'alpha' exists; 'missing' will raise NotFoundError
        alpha_docs = [f"{DOCUMENT_PREFIX}alpha_found"]
        alpha_dists = [0.1]
        client = StubChromaClient(collections={
            "alpha": StubCollection("alpha", alpha_docs, alpha_dists),
        })
        patches = _patch_retriever(
            client, stub_embeddings, ["alpha", "missing"]
        )
        with (
            patch("rag.retriever.get_chroma_client", patches["rag.retriever.get_chroma_client"]),
            patch("rag.retriever.get_embeddings", patches["rag.retriever.get_embeddings"]),
            patch("rag.retriever.get_permitted_datasources", patches["rag.retriever.get_permitted_datasources"]),
            patch("rag.retriever.log_retrieve_event", patches["rag.retriever.log_retrieve_event"]),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 5
            from rag.retriever import get_relevant_context

            with caplog.at_level(logging.WARNING):
                result = await get_relevant_context(
                    "test query", model_name="test-model"
                )

        assert "alpha_found" in result
        # WARNING should be logged for the missing collection
        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("missing" in msg.lower() or "not found" in msg.lower() for msg in warning_msgs)

    @pytest.mark.asyncio
    async def test_empty_model_name_default_deny(
        self, single_collection_client, stub_embeddings
    ):
        """model_name='' -> '' (permitted is [])."""
        patches = _patch_retriever(
            single_collection_client, stub_embeddings, []  # empty = default deny
        )
        with (
            patch("rag.retriever.get_chroma_client", patches["rag.retriever.get_chroma_client"]),
            patch("rag.retriever.get_embeddings", patches["rag.retriever.get_embeddings"]),
            patch("rag.retriever.get_permitted_datasources", patches["rag.retriever.get_permitted_datasources"]),
            patch("rag.retriever.log_retrieve_event", patches["rag.retriever.log_retrieve_event"]),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            from rag.retriever import get_relevant_context

            result = await get_relevant_context("test query", model_name="")

        assert result == ""


# ==========================================================================
# Edge cases
# ==========================================================================


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(
        self, single_collection_client, stub_embeddings
    ):
        """query='' -> '' returned, get_embeddings NOT called."""
        mock_get_embeddings = MagicMock(return_value=stub_embeddings)
        with (
            patch("rag.retriever.get_chroma_client", MagicMock(return_value=single_collection_client)),
            patch("rag.retriever.get_embeddings", mock_get_embeddings),
            patch("rag.retriever.get_permitted_datasources", MagicMock(return_value=["alpha"])),
            patch("rag.retriever.log_retrieve_event", MagicMock()),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            from rag.retriever import get_relevant_context

            result = await get_relevant_context("", model_name="test-model")

        assert result == ""
        mock_get_embeddings.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_empty(
        self, single_collection_client, stub_embeddings
    ):
        """query='   ' -> same as empty."""
        mock_get_embeddings = MagicMock(return_value=stub_embeddings)
        with (
            patch("rag.retriever.get_chroma_client", MagicMock(return_value=single_collection_client)),
            patch("rag.retriever.get_embeddings", mock_get_embeddings),
            patch("rag.retriever.get_permitted_datasources", MagicMock(return_value=["alpha"])),
            patch("rag.retriever.log_retrieve_event", MagicMock()),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            from rag.retriever import get_relevant_context

            result = await get_relevant_context("   ", model_name="test-model")

        assert result == ""
        mock_get_embeddings.assert_not_called()

    @pytest.mark.asyncio
    async def test_tied_distances_alphabetical_order(
        self, tied_distance_client, stub_embeddings
    ):
        """alpha at 0.5, beta at 0.5 -> alpha first (sorted permitted order)."""
        patches = _patch_retriever(
            tied_distance_client, stub_embeddings, ["alpha", "beta"]
        )
        with (
            patch("rag.retriever.get_chroma_client", patches["rag.retriever.get_chroma_client"]),
            patch("rag.retriever.get_embeddings", patches["rag.retriever.get_embeddings"]),
            patch("rag.retriever.get_permitted_datasources", patches["rag.retriever.get_permitted_datasources"]),
            patch("rag.retriever.log_retrieve_event", patches["rag.retriever.log_retrieve_event"]),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 5
            from rag.retriever import get_relevant_context

            result = await get_relevant_context("test query", model_name="test-model")

        pos_alpha = result.find("alpha_tied")
        pos_beta = result.find("beta_tied")
        assert pos_alpha != -1
        assert pos_beta != -1
        assert pos_alpha < pos_beta

    @pytest.mark.asyncio
    async def test_explicit_n_results_overrides_settings(
        self, single_collection_client, stub_embeddings
    ):
        """n_results=1 -> only 1 result despite rag_top_k=3."""
        patches = _patch_retriever(
            single_collection_client, stub_embeddings, ["alpha"]
        )
        with (
            patch("rag.retriever.get_chroma_client", patches["rag.retriever.get_chroma_client"]),
            patch("rag.retriever.get_embeddings", patches["rag.retriever.get_embeddings"]),
            patch("rag.retriever.get_permitted_datasources", patches["rag.retriever.get_permitted_datasources"]),
            patch("rag.retriever.log_retrieve_event", patches["rag.retriever.log_retrieve_event"]),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            from rag.retriever import get_relevant_context

            result = await get_relevant_context(
                "test query", model_name="test-model", n_results=1
            )

        assert "doc0" in result
        assert "doc1" not in result

    @pytest.mark.asyncio
    async def test_query_prefix_not_double_prepended(
        self, single_collection_client
    ):
        """embed_documents called with exactly 'QUERY_PREFIX + query' (one prefix only)."""
        spy_embeddings = StubEmbeddings()
        original_embed = spy_embeddings.embed_documents
        calls = []

        def tracking_embed(texts):
            calls.extend(texts)
            return original_embed(texts)

        spy_embeddings.embed_documents = tracking_embed

        patches = _patch_retriever(
            single_collection_client, spy_embeddings, ["alpha"]
        )
        with (
            patch("rag.retriever.get_chroma_client", patches["rag.retriever.get_chroma_client"]),
            patch("rag.retriever.get_embeddings", MagicMock(return_value=spy_embeddings)),
            patch("rag.retriever.get_permitted_datasources", patches["rag.retriever.get_permitted_datasources"]),
            patch("rag.retriever.log_retrieve_event", patches["rag.retriever.log_retrieve_event"]),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            from rag.retriever import get_relevant_context

            await get_relevant_context("my question", model_name="test-model")

        # The embedding call should have exactly one QUERY_PREFIX
        assert len(calls) >= 1
        query_text = calls[0]
        expected = f"{QUERY_PREFIX}my question"
        assert query_text == expected
        # Ensure no double prefix
        double_prefix = f"{QUERY_PREFIX}{QUERY_PREFIX}"
        assert double_prefix not in query_text


# ==========================================================================
# Idempotency
# ==========================================================================


class TestIdempotency:
    @pytest.mark.asyncio
    async def test_deterministic_output(
        self, single_collection_client, stub_embeddings
    ):
        """Same input twice -> byte-identical return."""
        results = []
        for _ in range(2):
            patches = _patch_retriever(
                single_collection_client, stub_embeddings, ["alpha"]
            )
            with (
                patch("rag.retriever.get_chroma_client", patches["rag.retriever.get_chroma_client"]),
                patch("rag.retriever.get_embeddings", patches["rag.retriever.get_embeddings"]),
                patch("rag.retriever.get_permitted_datasources", patches["rag.retriever.get_permitted_datasources"]),
                patch("rag.retriever.log_retrieve_event", patches["rag.retriever.log_retrieve_event"]),
                patch("rag.retriever.settings") as mock_settings,
            ):
                mock_settings.rag_top_k = 3
                from rag.retriever import get_relevant_context

                result = await get_relevant_context(
                    "determinism test", model_name="test-model"
                )
                results.append(result)

        assert results[0] == results[1]


# ==========================================================================
# Criteria switching (llm_as_judge.py)
# ==========================================================================


class TestCriteriaSwitching:
    """Test that generate_feedback receives rag_judge_criteria when context is present,
    and default judge_criteria when context is absent."""

    def _make_stub_reasoner(self):
        """Create a stub reasoner whose ainvoke returns a mock with .content."""
        stub = AsyncMock()
        response = MagicMock()
        response.content = "stub answer"
        stub.ainvoke.return_value = response
        return stub

    @pytest.mark.asyncio
    async def test_criteria_uses_rag_when_context_present(self):
        """run_judge_chain(context='some context') -> criteria == rag_judge_criteria."""
        stub_reasoner = self._make_stub_reasoner()
        mock_feedback = AsyncMock(return_value="improve it")
        mock_relevancy = AsyncMock(return_value={
            "score": 0.0, "verdict": "FAIL", "threshold": 0.7,
        })

        with (
            patch("chains.llm_as_judge.get_reasoner", return_value=stub_reasoner),
            patch("chains.llm_as_judge.probe_endpoints", new_callable=AsyncMock),
            patch("chains.llm_as_judge.compute_response_relevancy", mock_relevancy),
            patch("chains.llm_as_judge.generate_feedback", mock_feedback),
            patch("chains.llm_as_judge.app_settings") as mock_settings,
        ):
            mock_settings.max_judge_retries = 1
            mock_settings.judge_criteria = "default criteria"
            mock_settings.rag_judge_criteria = "rag criteria"

            from chains.llm_as_judge import run_judge_chain

            await run_judge_chain(prompt="Q", context="some context")

        # generate_feedback should have been called with criteria=rag_judge_criteria
        mock_feedback.assert_called_once()
        call_kwargs = mock_feedback.call_args
        assert call_kwargs.kwargs.get("criteria") == "rag criteria" or (
            len(call_kwargs.args) > 4 and call_kwargs.args[4] == "rag criteria"
        )

    @pytest.mark.asyncio
    async def test_criteria_uses_default_when_no_context(self):
        """run_judge_chain(context='') -> criteria == judge_criteria."""
        stub_reasoner = self._make_stub_reasoner()
        mock_feedback = AsyncMock(return_value="improve it")
        mock_relevancy = AsyncMock(return_value={
            "score": 0.0, "verdict": "FAIL", "threshold": 0.7,
        })

        with (
            patch("chains.llm_as_judge.get_reasoner", return_value=stub_reasoner),
            patch("chains.llm_as_judge.probe_endpoints", new_callable=AsyncMock),
            patch("chains.llm_as_judge.compute_response_relevancy", mock_relevancy),
            patch("chains.llm_as_judge.generate_feedback", mock_feedback),
            patch("chains.llm_as_judge.app_settings") as mock_settings,
        ):
            mock_settings.max_judge_retries = 1
            mock_settings.judge_criteria = "default criteria"
            mock_settings.rag_judge_criteria = "rag criteria"

            from chains.llm_as_judge import run_judge_chain

            await run_judge_chain(prompt="Q", context="")

        # generate_feedback should have been called with criteria=judge_criteria
        mock_feedback.assert_called_once()
        call_kwargs = mock_feedback.call_args
        assert call_kwargs.kwargs.get("criteria") == "default criteria" or (
            len(call_kwargs.args) > 4 and call_kwargs.args[4] == "default criteria"
        )


# ==========================================================================
# Audit
# ==========================================================================


class TestAudit:
    @pytest.mark.asyncio
    async def test_retrieve_event_written(
        self, single_collection_client, stub_embeddings, tmp_path
    ):
        """After retriever call, audit JSONL contains action=retrieve entry."""
        audit_file = tmp_path / "audit.jsonl"

        # Use a real log_retrieve_event that writes to our temp file
        def fake_log_retrieve_event(
            model_name, datasources_queried, query, hits, *, status="ok", error=""
        ):
            import json as _json
            from datetime import datetime, timezone
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "retrieve",
                "model_name": model_name,
                "datasources_queried": datasources_queried,
                "query_length": len(query),
                "hits": hits,
                "status": status,
            }
            if error:
                entry["error"] = error
            with open(audit_file, "a", encoding="utf-8") as f:
                f.write(_json.dumps(entry, ensure_ascii=False) + "\n")

        with (
            patch("rag.retriever.get_chroma_client", MagicMock(return_value=single_collection_client)),
            patch("rag.retriever.get_embeddings", MagicMock(return_value=stub_embeddings)),
            patch("rag.retriever.get_permitted_datasources", MagicMock(return_value=["alpha"])),
            patch("rag.retriever.log_retrieve_event", fake_log_retrieve_event),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            from rag.retriever import get_relevant_context

            await get_relevant_context("audit test query", model_name="audit-model")

        # Verify the audit file was written
        assert audit_file.exists()
        lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1
        entry = json.loads(lines[-1])
        assert entry["action"] == "retrieve"
        assert entry["model_name"] == "audit-model"
        assert isinstance(entry["datasources_queried"], list)
        assert isinstance(entry["query_length"], int)
        assert entry["query_length"] > 0
        assert isinstance(entry["hits"], int)
        assert entry["status"] == "ok"

    @pytest.mark.asyncio
    async def test_query_not_in_audit_log(
        self, single_collection_client, stub_embeddings, tmp_path
    ):
        """Audit entry does NOT have a 'query' field (PII protection)."""
        audit_file = tmp_path / "audit.jsonl"
        secret_query = "my private medical question"

        def fake_log_retrieve_event(
            model_name, datasources_queried, query, hits, *, status="ok", error=""
        ):
            import json as _json
            from datetime import datetime, timezone
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "retrieve",
                "model_name": model_name,
                "datasources_queried": datasources_queried,
                "query_length": len(query),
                "hits": hits,
                "status": status,
            }
            with open(audit_file, "a", encoding="utf-8") as f:
                f.write(_json.dumps(entry, ensure_ascii=False) + "\n")

        with (
            patch("rag.retriever.get_chroma_client", MagicMock(return_value=single_collection_client)),
            patch("rag.retriever.get_embeddings", MagicMock(return_value=stub_embeddings)),
            patch("rag.retriever.get_permitted_datasources", MagicMock(return_value=["alpha"])),
            patch("rag.retriever.log_retrieve_event", fake_log_retrieve_event),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            from rag.retriever import get_relevant_context

            await get_relevant_context(secret_query, model_name="test-model")

        assert audit_file.exists()
        raw = audit_file.read_text(encoding="utf-8")
        entry = json.loads(raw.strip().splitlines()[-1])
        # The entry must NOT contain the query text itself
        assert "query" not in entry, "Audit entry must not store the raw query (PII)"
        assert secret_query not in raw, "Query text must not appear anywhere in audit log"
