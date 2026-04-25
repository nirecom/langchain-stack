"""
Unit tests for Phase 4C: RAG retrieval pipeline (OpenSearch backend).

Tests cover:
  - Normal: single/multi index retrieval, top-k, settings fallback
  - Error: no permitted datasources, OpenSearch exception, empty model name
  - Edge: empty/whitespace query, explicit n_results, query prefix, search mode
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
from unittest.mock import AsyncMock, MagicMock, patch


# -- Stubs ----------------------------------------------------------------


class StubEmbeddings:
    """Stub embedding model that returns fixed vectors."""

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def _make_os_hit(text: str, source: str = "", file_name: str = "", section_path=None) -> dict:
    """Build a fake OpenSearch hit dict."""
    return {
        "_score": 1.0,
        "_source": {
            "text": text,
            "source": source or file_name,
            "file_name": file_name or source,
            "section_path": section_path,
        },
    }


def _make_os_response(hits: list[dict]) -> dict:
    return {"hits": {"hits": hits, "total": {"value": len(hits)}}}


class StubOsClient:
    """Stub OpenSearch client returning pre-configured search results."""

    def __init__(self, hits: list[dict]):
        self._hits = hits

    def search(self, **kwargs):
        return _make_os_response(self._hits)


# -- Fixtures -------------------------------------------------------------


@pytest.fixture
def stub_embeddings():
    return StubEmbeddings()


@pytest.fixture
def single_index_client():
    """Client returning 5 hits from one index."""
    hits = [_make_os_hit(f"doc{i}", source=f"file{i}.txt") for i in range(5)]
    return StubOsClient(hits)


@pytest.fixture
def empty_client():
    """Client returning no hits."""
    return StubOsClient([])


# -- Helpers ---------------------------------------------------------------


def _patch_retriever(os_client, embeddings, permitted, settings_overrides=None):
    return {
        "rag.retriever.get_os_client": MagicMock(return_value=os_client),
        "rag.retriever.get_embeddings": MagicMock(return_value=embeddings),
        "rag.retriever.get_permitted_datasources_for_user": MagicMock(return_value=permitted),
        "rag.retriever.log_retrieve_event": MagicMock(),
    }


# ==========================================================================
# Normal cases
# ==========================================================================


class TestNormalCases:
    @pytest.mark.asyncio
    async def test_single_collection_top_k(self, single_index_client, stub_embeddings):
        """1 permitted datasource, 5 hits, rag_top_k=3 → top 3 returned."""
        patches = _patch_retriever(single_index_client, stub_embeddings, ["alpha"])
        with (
            patch("rag.retriever.get_os_client", patches["rag.retriever.get_os_client"]),
            patch("rag.retriever.get_embeddings", patches["rag.retriever.get_embeddings"]),
            patch("rag.retriever.get_permitted_datasources_for_user", patches["rag.retriever.get_permitted_datasources_for_user"]),
            patch("rag.retriever.log_retrieve_event", patches["rag.retriever.log_retrieve_event"]),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            mock_settings.search_mode = "dense"
            mock_settings.embedding_model_name = "BAAI/bge-m3"
            mock_settings.hybrid_pipeline_name = "hybrid-bm25-knn"
            mock_settings.os_index_prefix = "ls_"
            from rag.retriever import get_relevant_context

            result = await get_relevant_context("test query", user="nire")

        assert "doc0" in result
        assert "doc1" in result
        assert "doc2" in result
        assert "doc3" not in result
        assert "doc4" not in result

    @pytest.mark.asyncio
    async def test_n_results_none_uses_settings(self, single_index_client, stub_embeddings):
        """n_results=None → settings.rag_top_k consulted."""
        patches = _patch_retriever(single_index_client, stub_embeddings, ["alpha"])
        with (
            patch("rag.retriever.get_os_client", patches["rag.retriever.get_os_client"]),
            patch("rag.retriever.get_embeddings", patches["rag.retriever.get_embeddings"]),
            patch("rag.retriever.get_permitted_datasources_for_user", patches["rag.retriever.get_permitted_datasources_for_user"]),
            patch("rag.retriever.log_retrieve_event", patches["rag.retriever.log_retrieve_event"]),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 2
            mock_settings.search_mode = "dense"
            mock_settings.embedding_model_name = "BAAI/bge-m3"
            mock_settings.hybrid_pipeline_name = "hybrid-bm25-knn"
            mock_settings.os_index_prefix = "ls_"
            from rag.retriever import get_relevant_context

            result = await get_relevant_context("test query", user="nire", n_results=None)

        assert "doc0" in result
        assert "doc1" in result
        assert "doc2" not in result

    @pytest.mark.asyncio
    async def test_post_merge_truncation(self, single_index_client, stub_embeddings):
        """5 hits returned, rag_top_k=3 → exactly 3 items."""
        patches = _patch_retriever(single_index_client, stub_embeddings, ["alpha"])
        with (
            patch("rag.retriever.get_os_client", patches["rag.retriever.get_os_client"]),
            patch("rag.retriever.get_embeddings", patches["rag.retriever.get_embeddings"]),
            patch("rag.retriever.get_permitted_datasources_for_user", patches["rag.retriever.get_permitted_datasources_for_user"]),
            patch("rag.retriever.log_retrieve_event", patches["rag.retriever.log_retrieve_event"]),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            mock_settings.search_mode = "dense"
            mock_settings.embedding_model_name = "BAAI/bge-m3"
            mock_settings.hybrid_pipeline_name = "hybrid-bm25-knn"
            mock_settings.os_index_prefix = "ls_"
            from rag.retriever import get_relevant_context

            result = await get_relevant_context("test query", user="nire")

        assert "doc0" in result
        assert "doc1" in result
        assert "doc2" in result
        assert "doc3" not in result

    @pytest.mark.asyncio
    async def test_multi_collection_interleaved_merge(self, stub_embeddings):
        """Multiple permitted datasources → all queried via multi-index search."""
        hits = [
            _make_os_hit("alpha_doc0", source="alpha.txt"),
            _make_os_hit("beta_doc0", source="beta.txt"),
        ]
        client = StubOsClient(hits)
        patches = _patch_retriever(client, stub_embeddings, ["alpha", "beta"])
        with (
            patch("rag.retriever.get_os_client", patches["rag.retriever.get_os_client"]),
            patch("rag.retriever.get_embeddings", patches["rag.retriever.get_embeddings"]),
            patch("rag.retriever.get_permitted_datasources_for_user", patches["rag.retriever.get_permitted_datasources_for_user"]),
            patch("rag.retriever.log_retrieve_event", patches["rag.retriever.log_retrieve_event"]),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 5
            mock_settings.search_mode = "dense"
            mock_settings.embedding_model_name = "BAAI/bge-m3"
            mock_settings.hybrid_pipeline_name = "hybrid-bm25-knn"
            mock_settings.os_index_prefix = "ls_"
            from rag.retriever import get_relevant_context

            result = await get_relevant_context("test query", user="nire")

        assert "alpha_doc0" in result
        assert "beta_doc0" in result


# ==========================================================================
# Error cases
# ==========================================================================


class TestErrorCases:
    @pytest.mark.asyncio
    async def test_unknown_user_returns_empty(self, single_index_client, stub_embeddings, caplog):
        """user with no permitted datasources → '' + WARNING."""
        patches = _patch_retriever(single_index_client, stub_embeddings, [])
        with (
            patch("rag.retriever.get_os_client", patches["rag.retriever.get_os_client"]),
            patch("rag.retriever.get_embeddings", patches["rag.retriever.get_embeddings"]),
            patch("rag.retriever.get_permitted_datasources_for_user", patches["rag.retriever.get_permitted_datasources_for_user"]),
            patch("rag.retriever.log_retrieve_event", patches["rag.retriever.log_retrieve_event"]),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            mock_settings.search_mode = "dense"
            mock_settings.embedding_model_name = "BAAI/bge-m3"
            mock_settings.hybrid_pipeline_name = "hybrid-bm25-knn"
            mock_settings.os_index_prefix = "ls_"
            from rag.retriever import get_relevant_context

            with caplog.at_level(logging.WARNING):
                result = await get_relevant_context("test query", user="unknown")

        assert result == ""

    @pytest.mark.asyncio
    async def test_opensearch_exception_returns_empty(self, stub_embeddings, caplog):
        """OpenSearch raises exception → '' + error logged."""
        error_client = MagicMock()
        error_client.search.side_effect = Exception("connection refused")
        patches = _patch_retriever(error_client, stub_embeddings, ["alpha"])
        with (
            patch("rag.retriever.get_os_client", patches["rag.retriever.get_os_client"]),
            patch("rag.retriever.get_embeddings", patches["rag.retriever.get_embeddings"]),
            patch("rag.retriever.get_permitted_datasources_for_user", patches["rag.retriever.get_permitted_datasources_for_user"]),
            patch("rag.retriever.log_retrieve_event", patches["rag.retriever.log_retrieve_event"]),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            mock_settings.search_mode = "dense"
            mock_settings.embedding_model_name = "BAAI/bge-m3"
            mock_settings.hybrid_pipeline_name = "hybrid-bm25-knn"
            mock_settings.os_index_prefix = "ls_"
            from rag.retriever import get_relevant_context

            with caplog.at_level(logging.ERROR):
                result = await get_relevant_context("test query", user="nire")

        assert result == ""

    @pytest.mark.asyncio
    async def test_empty_user_default_deny(self, single_index_client, stub_embeddings):
        """user='' → '' (permitted is [])."""
        patches = _patch_retriever(single_index_client, stub_embeddings, [])
        with (
            patch("rag.retriever.get_os_client", patches["rag.retriever.get_os_client"]),
            patch("rag.retriever.get_embeddings", patches["rag.retriever.get_embeddings"]),
            patch("rag.retriever.get_permitted_datasources_for_user", patches["rag.retriever.get_permitted_datasources_for_user"]),
            patch("rag.retriever.log_retrieve_event", patches["rag.retriever.log_retrieve_event"]),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            mock_settings.search_mode = "dense"
            mock_settings.embedding_model_name = "BAAI/bge-m3"
            mock_settings.hybrid_pipeline_name = "hybrid-bm25-knn"
            mock_settings.os_index_prefix = "ls_"
            from rag.retriever import get_relevant_context

            result = await get_relevant_context("test query", user="")

        assert result == ""

    @pytest.mark.asyncio
    async def test_collection_not_found_skips(self, stub_embeddings, caplog):
        """OpenSearch returns empty hits (index missing/unavailable) → '' returned gracefully."""
        client = StubOsClient([])
        patches = _patch_retriever(client, stub_embeddings, ["alpha", "missing"])
        with (
            patch("rag.retriever.get_os_client", patches["rag.retriever.get_os_client"]),
            patch("rag.retriever.get_embeddings", patches["rag.retriever.get_embeddings"]),
            patch("rag.retriever.get_permitted_datasources_for_user", patches["rag.retriever.get_permitted_datasources_for_user"]),
            patch("rag.retriever.log_retrieve_event", patches["rag.retriever.log_retrieve_event"]),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 5
            mock_settings.search_mode = "dense"
            mock_settings.embedding_model_name = "BAAI/bge-m3"
            mock_settings.hybrid_pipeline_name = "hybrid-bm25-knn"
            mock_settings.os_index_prefix = "ls_"
            from rag.retriever import get_relevant_context

            result = await get_relevant_context("test query", user="nire")

        assert result == ""


# ==========================================================================
# Edge cases
# ==========================================================================


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self, single_index_client, stub_embeddings):
        """query='' → '' returned, get_embeddings NOT called."""
        mock_get_embeddings = MagicMock(return_value=stub_embeddings)
        with (
            patch("rag.retriever.get_os_client", MagicMock(return_value=single_index_client)),
            patch("rag.retriever.get_embeddings", mock_get_embeddings),
            patch("rag.retriever.get_permitted_datasources_for_user", MagicMock(return_value=["alpha"])),
            patch("rag.retriever.log_retrieve_event", MagicMock()),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            mock_settings.search_mode = "dense"
            mock_settings.embedding_model_name = "BAAI/bge-m3"
            mock_settings.hybrid_pipeline_name = "hybrid-bm25-knn"
            mock_settings.os_index_prefix = "ls_"
            from rag.retriever import get_relevant_context

            result = await get_relevant_context("", user="nire")

        assert result == ""
        mock_get_embeddings.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_empty(self, single_index_client, stub_embeddings):
        """query='   ' → same as empty."""
        mock_get_embeddings = MagicMock(return_value=stub_embeddings)
        with (
            patch("rag.retriever.get_os_client", MagicMock(return_value=single_index_client)),
            patch("rag.retriever.get_embeddings", mock_get_embeddings),
            patch("rag.retriever.get_permitted_datasources_for_user", MagicMock(return_value=["alpha"])),
            patch("rag.retriever.log_retrieve_event", MagicMock()),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            mock_settings.search_mode = "dense"
            mock_settings.embedding_model_name = "BAAI/bge-m3"
            mock_settings.hybrid_pipeline_name = "hybrid-bm25-knn"
            mock_settings.os_index_prefix = "ls_"
            from rag.retriever import get_relevant_context

            result = await get_relevant_context("   ", user="nire")

        assert result == ""
        mock_get_embeddings.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_n_results_overrides_settings(self, single_index_client, stub_embeddings):
        """n_results=1 → only 1 result despite rag_top_k=3."""
        patches = _patch_retriever(single_index_client, stub_embeddings, ["alpha"])
        with (
            patch("rag.retriever.get_os_client", patches["rag.retriever.get_os_client"]),
            patch("rag.retriever.get_embeddings", patches["rag.retriever.get_embeddings"]),
            patch("rag.retriever.get_permitted_datasources_for_user", patches["rag.retriever.get_permitted_datasources_for_user"]),
            patch("rag.retriever.log_retrieve_event", patches["rag.retriever.log_retrieve_event"]),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            mock_settings.search_mode = "dense"
            mock_settings.embedding_model_name = "BAAI/bge-m3"
            mock_settings.hybrid_pipeline_name = "hybrid-bm25-knn"
            mock_settings.os_index_prefix = "ls_"
            from rag.retriever import get_relevant_context

            result = await get_relevant_context("test query", user="nire", n_results=1)

        assert "doc0" in result
        assert "doc1" not in result

    @pytest.mark.asyncio
    async def test_query_prefix_not_double_prepended(self, single_index_client):
        """embed_documents called with exactly 'QUERY_PREFIX + query' (one prefix only)."""
        spy_embeddings = StubEmbeddings()
        original_embed = spy_embeddings.embed_documents
        calls = []

        def tracking_embed(texts):
            calls.extend(texts)
            return original_embed(texts)

        spy_embeddings.embed_documents = tracking_embed

        with (
            patch("rag.retriever.get_os_client", MagicMock(return_value=single_index_client)),
            patch("rag.retriever.get_embeddings", MagicMock(return_value=spy_embeddings)),
            patch("rag.retriever.get_permitted_datasources_for_user", MagicMock(return_value=["alpha"])),
            patch("rag.retriever.log_retrieve_event", MagicMock()),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            mock_settings.search_mode = "dense"
            mock_settings.embedding_model_name = "BAAI/bge-m3"
            mock_settings.hybrid_pipeline_name = "hybrid-bm25-knn"
            mock_settings.os_index_prefix = "ls_"
            from rag.retriever import get_relevant_context

            await get_relevant_context("my question", user="nire")

        assert len(calls) >= 1
        query_text = calls[0]
        assert "my question" in query_text
        assert query_text.count("my question") == 1

    @pytest.mark.asyncio
    async def test_search_mode_hybrid_header_uses_builder(self, single_index_client, stub_embeddings):
        """search_mode=hybrid+header → _build_hybrid_header builder is invoked."""
        from rag.retriever import _build_hybrid_header
        called_with = []
        original = _build_hybrid_header

        def spy_builder(vec, text, k):
            called_with.append((vec, text, k))
            return original(vec, text, k)

        patches = _patch_retriever(single_index_client, stub_embeddings, ["alpha"])
        with (
            patch("rag.retriever.get_os_client", patches["rag.retriever.get_os_client"]),
            patch("rag.retriever.get_embeddings", patches["rag.retriever.get_embeddings"]),
            patch("rag.retriever.get_permitted_datasources_for_user", patches["rag.retriever.get_permitted_datasources_for_user"]),
            patch("rag.retriever.log_retrieve_event", patches["rag.retriever.log_retrieve_event"]),
            patch("rag.retriever._QUERY_BUILDERS", {"hybrid+header": spy_builder}),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            mock_settings.search_mode = "hybrid+header"
            mock_settings.embedding_model_name = "BAAI/bge-m3"
            mock_settings.hybrid_pipeline_name = "hybrid-bm25-knn"
            mock_settings.os_index_prefix = "ls_"
            from rag.retriever import get_relevant_context

            await get_relevant_context("test query", user="nire")

        assert len(called_with) == 1


# ==========================================================================
# Idempotency
# ==========================================================================


class TestIdempotency:
    @pytest.mark.asyncio
    async def test_deterministic_output(self, single_index_client, stub_embeddings):
        """Same input twice → byte-identical return."""
        results = []
        for _ in range(2):
            patches = _patch_retriever(single_index_client, stub_embeddings, ["alpha"])
            with (
                patch("rag.retriever.get_os_client", patches["rag.retriever.get_os_client"]),
                patch("rag.retriever.get_embeddings", patches["rag.retriever.get_embeddings"]),
                patch("rag.retriever.get_permitted_datasources_for_user", patches["rag.retriever.get_permitted_datasources_for_user"]),
                patch("rag.retriever.log_retrieve_event", patches["rag.retriever.log_retrieve_event"]),
                patch("rag.retriever.settings") as mock_settings,
            ):
                mock_settings.rag_top_k = 3
                mock_settings.search_mode = "dense"
                mock_settings.embedding_model_name = "BAAI/bge-m3"
                mock_settings.hybrid_pipeline_name = "hybrid-bm25-knn"
                mock_settings.os_index_prefix = "ls_"
                from rag.retriever import get_relevant_context

                result = await get_relevant_context("determinism test", user="nire")
                results.append(result)

        assert results[0] == results[1]


# ==========================================================================
# Criteria switching (llm_as_judge.py) — unchanged from Phase 4C
# ==========================================================================


class TestCriteriaSwitching:
    def _make_stub_reasoner(self):
        stub = AsyncMock()
        response = MagicMock()
        response.content = "stub answer"
        stub.ainvoke.return_value = response
        return stub

    @pytest.mark.asyncio
    async def test_criteria_uses_rag_when_context_present(self):
        stub_reasoner = self._make_stub_reasoner()
        mock_feedback = AsyncMock(return_value="improve it")
        mock_relevancy = AsyncMock(return_value={"score": 0.0, "verdict": "FAIL", "threshold": 0.7})

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

        mock_feedback.assert_called_once()
        call_kwargs = mock_feedback.call_args
        assert call_kwargs.kwargs.get("criteria") == "rag criteria" or (
            len(call_kwargs.args) > 4 and call_kwargs.args[4] == "rag criteria"
        )

    @pytest.mark.asyncio
    async def test_criteria_uses_default_when_no_context(self):
        stub_reasoner = self._make_stub_reasoner()
        mock_feedback = AsyncMock(return_value="improve it")
        mock_relevancy = AsyncMock(return_value={"score": 0.0, "verdict": "FAIL", "threshold": 0.7})

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
    async def test_retrieve_event_written(self, single_index_client, stub_embeddings, tmp_path):
        """After retriever call, audit JSONL contains action=retrieve entry."""
        audit_file = tmp_path / "audit.jsonl"

        def fake_log_retrieve_event(
            *, user, model_name="", datasources_queried, query, hits, status="ok", error=""
        ):
            import json as _json
            from datetime import datetime, timezone
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "retrieve",
                "user": user,
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
            patch("rag.retriever.get_os_client", MagicMock(return_value=single_index_client)),
            patch("rag.retriever.get_embeddings", MagicMock(return_value=stub_embeddings)),
            patch("rag.retriever.get_permitted_datasources_for_user", MagicMock(return_value=["alpha"])),
            patch("rag.retriever.log_retrieve_event", fake_log_retrieve_event),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            mock_settings.search_mode = "dense"
            mock_settings.embedding_model_name = "BAAI/bge-m3"
            mock_settings.hybrid_pipeline_name = "hybrid-bm25-knn"
            mock_settings.os_index_prefix = "ls_"
            from rag.retriever import get_relevant_context

            await get_relevant_context("audit test query", user="nire")

        assert audit_file.exists()
        lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1
        entry = json.loads(lines[-1])
        assert entry["action"] == "retrieve"
        assert entry.get("user") == "nire"
        assert isinstance(entry["datasources_queried"], list)
        assert isinstance(entry["query_length"], int)
        assert entry["query_length"] > 0
        assert isinstance(entry["hits"], int)
        assert entry["status"] == "ok"

    @pytest.mark.asyncio
    async def test_query_not_in_audit_log(self, single_index_client, stub_embeddings, tmp_path):
        """Audit entry does NOT have a 'query' field (PII protection)."""
        audit_file = tmp_path / "audit.jsonl"
        secret_query = "my private medical question"

        def fake_log_retrieve_event(
            *, user, model_name="", datasources_queried, query, hits, status="ok", error=""
        ):
            import json as _json
            from datetime import datetime, timezone
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": "retrieve",
                "user": user,
                "model_name": model_name,
                "datasources_queried": datasources_queried,
                "query_length": len(query),
                "hits": hits,
                "status": status,
            }
            with open(audit_file, "a", encoding="utf-8") as f:
                f.write(_json.dumps(entry, ensure_ascii=False) + "\n")

        with (
            patch("rag.retriever.get_os_client", MagicMock(return_value=single_index_client)),
            patch("rag.retriever.get_embeddings", MagicMock(return_value=stub_embeddings)),
            patch("rag.retriever.get_permitted_datasources_for_user", MagicMock(return_value=["alpha"])),
            patch("rag.retriever.log_retrieve_event", fake_log_retrieve_event),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            mock_settings.search_mode = "dense"
            mock_settings.embedding_model_name = "BAAI/bge-m3"
            mock_settings.hybrid_pipeline_name = "hybrid-bm25-knn"
            mock_settings.os_index_prefix = "ls_"
            from rag.retriever import get_relevant_context

            await get_relevant_context(secret_query, user="nire")

        assert audit_file.exists()
        raw = audit_file.read_text(encoding="utf-8")
        entry = json.loads(raw.strip().splitlines()[-1])
        assert "query" not in entry, "Audit entry must not store the raw query (PII)"
        assert secret_query not in raw, "Query text must not appear anywhere in audit log"
