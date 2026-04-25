"""
Unit tests for embedding adapter layer (TDD — implementation not yet written).

Covers:
  A. embedding_adapters.py — get_adapter() model-to-config mapping
  B. embeddings.py — get_embeddings(role="query"|"ingest") with per-role cache
  C. evaluation/metrics.py — compute_faithfulness() / compute_context_precision()

All tests are designed to fail with ImportError / AttributeError until the
corresponding implementation is added. No test logic bugs are intentional.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import logging
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call


# ==========================================================================
# A. embedding_adapters.py
# ==========================================================================

class TestGetAdapterNormal:
    """Normal cases: known models return correct EmbeddingAdapter instances."""

    def test_ruri_v3_prefixes(self):
        """ruri-v3-310m -> document_prefix='検索文書: ', query_prefix='検索クエリ: ', normalize=True."""
        from models.embedding_adapters import get_adapter
        adapter = get_adapter("cl-nagoya/ruri-v3-310m")
        assert adapter.model_name == "cl-nagoya/ruri-v3-310m"
        assert adapter.document_prefix == "検索文書: "
        assert adapter.query_prefix == "検索クエリ: "
        assert adapter.normalize is True

    def test_bge_m3_empty_prefixes(self):
        """BGE-M3 uses no prefix for either role, normalize=True."""
        from models.embedding_adapters import get_adapter
        adapter = get_adapter("BAAI/bge-m3")
        assert adapter.model_name == "BAAI/bge-m3"
        assert adapter.document_prefix == ""
        assert adapter.query_prefix == ""
        assert adapter.normalize is True

    def test_qwen3_embedding_instruct_query_prefix(self):
        """Qwen3-Embedding -> document_prefix='', query_prefix is non-empty English instruct string, normalize=True."""
        from models.embedding_adapters import get_adapter
        adapter = get_adapter("Qwen/Qwen3-Embedding-0.6B")
        assert adapter.model_name == "Qwen/Qwen3-Embedding-0.6B"
        assert adapter.document_prefix == ""
        # query_prefix must be non-empty (instruct format)
        assert isinstance(adapter.query_prefix, str)
        assert len(adapter.query_prefix) > 0
        assert adapter.normalize is True

    def test_jina_v3_task_query_prefix(self):
        """jina-embeddings-v3 -> document_prefix='', query_prefix is non-empty jina task string, normalize=True."""
        from models.embedding_adapters import get_adapter
        adapter = get_adapter("jinaai/jina-embeddings-v3")
        assert adapter.model_name == "jinaai/jina-embeddings-v3"
        assert adapter.document_prefix == ""
        # query_prefix must be non-empty (task format)
        assert isinstance(adapter.query_prefix, str)
        assert len(adapter.query_prefix) > 0
        assert adapter.normalize is True


class TestGetAdapterError:
    """Error cases: unknown or invalid model names raise ValueError."""

    def test_unknown_model_raises_value_error(self):
        """get_adapter('unknown/model') -> ValueError."""
        from models.embedding_adapters import get_adapter
        with pytest.raises(ValueError):
            get_adapter("unknown/model")

    def test_empty_string_raises_value_error(self):
        """get_adapter('') -> ValueError."""
        from models.embedding_adapters import get_adapter
        with pytest.raises(ValueError):
            get_adapter("")


class TestEmbeddingAdapterDataclass:
    """Verify EmbeddingAdapter is a proper dataclass with expected fields."""

    def test_adapter_has_required_fields(self):
        """EmbeddingAdapter dataclass exposes model_name, document_prefix, query_prefix, normalize."""
        from models.embedding_adapters import EmbeddingAdapter
        adapter = EmbeddingAdapter(
            model_name="test/model",
            document_prefix="doc: ",
            query_prefix="query: ",
            normalize=True,
            dimension=768,
        )
        assert adapter.model_name == "test/model"
        assert adapter.document_prefix == "doc: "
        assert adapter.query_prefix == "query: "
        assert adapter.normalize is True


# ==========================================================================
# B. get_embeddings(role) — per-role cache
# ==========================================================================

@pytest.fixture(autouse=False)
def reset_embeddings_cache():
    """Reset module-level _query_embeddings / _ingest_embeddings between tests.

    langchain_community is not installed in the test environment, so we mock it
    at the sys.modules level before importing models.embeddings.
    """
    import sys
    import types

    # Stub out langchain_community so embeddings.py can be imported
    lc_mod = types.ModuleType("langchain_community")
    lc_emb = types.ModuleType("langchain_community.embeddings")
    lc_emb.HuggingFaceEmbeddings = MagicMock
    lc_mod.embeddings = lc_emb
    sys.modules.setdefault("langchain_community", lc_mod)
    sys.modules.setdefault("langchain_community.embeddings", lc_emb)

    # Also stub settings so the module-level import of settings works
    settings_stub = types.ModuleType("settings")
    settings_obj = MagicMock()
    settings_obj.embedding_model_name = "cl-nagoya/ruri-v3-310m"
    settings_obj.ingest_device = "cpu"
    settings_stub.settings = settings_obj
    sys.modules.setdefault("settings", settings_stub)

    # Now import (or reload) models.embeddings with stubs in place
    import importlib
    if "models.embeddings" in sys.modules:
        emb_mod = sys.modules["models.embeddings"]
    else:
        emb_mod = importlib.import_module("models.embeddings")

    # Reset before test
    emb_mod._query_embeddings = None
    emb_mod._ingest_embeddings = None
    yield
    # Reset after test
    emb_mod._query_embeddings = None
    emb_mod._ingest_embeddings = None


class TestGetEmbeddingsRoleNormal:
    """Normal cases for get_embeddings(role=...)."""

    def test_query_role_uses_cpu(self, reset_embeddings_cache):
        """role='query' -> HuggingFaceEmbeddings initialized with device='cpu'."""
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        with patch("models.embeddings.HuggingFaceEmbeddings", mock_cls):
            with patch("models.embeddings.settings") as mock_settings:
                mock_settings.embedding_model_name = "cl-nagoya/ruri-v3-310m"
                mock_settings.ingest_device = "cpu"

                from models.embeddings import get_embeddings
                result = get_embeddings(role="query")

        assert result is mock_instance
        init_kwargs = mock_cls.call_args.kwargs
        assert init_kwargs["model_kwargs"]["device"] == "cpu"

    def test_ingest_role_uses_cuda_when_configured(self, reset_embeddings_cache):
        """role='ingest', settings.ingest_device='cuda' -> device='cuda'."""
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        with patch("models.embeddings.HuggingFaceEmbeddings", mock_cls):
            with patch("models.embeddings.settings") as mock_settings:
                mock_settings.embedding_model_name = "cl-nagoya/ruri-v3-310m"
                mock_settings.ingest_device = "cuda"

                from models.embeddings import get_embeddings
                result = get_embeddings(role="ingest")

        assert result is mock_instance
        init_kwargs = mock_cls.call_args.kwargs
        assert init_kwargs["model_kwargs"]["device"] == "cuda"

    def test_ingest_role_defaults_to_cpu(self, reset_embeddings_cache):
        """role='ingest', settings.ingest_device='cpu' (default) -> device='cpu'."""
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        with patch("models.embeddings.HuggingFaceEmbeddings", mock_cls):
            with patch("models.embeddings.settings") as mock_settings:
                mock_settings.embedding_model_name = "cl-nagoya/ruri-v3-310m"
                mock_settings.ingest_device = "cpu"

                from models.embeddings import get_embeddings
                result = get_embeddings(role="ingest")

        assert result is mock_instance
        init_kwargs = mock_cls.call_args.kwargs
        assert init_kwargs["model_kwargs"]["device"] == "cpu"

    def test_query_role_singleton(self, reset_embeddings_cache):
        """Calling get_embeddings('query') twice returns the same object; __init__ called once."""
        mock_cls = MagicMock()
        instance_a = MagicMock()
        mock_cls.return_value = instance_a

        with patch("models.embeddings.HuggingFaceEmbeddings", mock_cls):
            with patch("models.embeddings.settings") as mock_settings:
                mock_settings.embedding_model_name = "cl-nagoya/ruri-v3-310m"
                mock_settings.ingest_device = "cpu"

                from models.embeddings import get_embeddings
                first = get_embeddings(role="query")
                second = get_embeddings(role="query")

        assert first is second
        assert mock_cls.call_count == 1

    def test_query_and_ingest_are_different_objects(self, reset_embeddings_cache):
        """get_embeddings('query') is not get_embeddings('ingest')."""
        query_instance = MagicMock(name="query_instance")
        ingest_instance = MagicMock(name="ingest_instance")
        call_count = {"n": 0}

        def side_effect(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return query_instance
            return ingest_instance

        mock_cls = MagicMock(side_effect=side_effect)

        with patch("models.embeddings.HuggingFaceEmbeddings", mock_cls):
            with patch("models.embeddings.settings") as mock_settings:
                mock_settings.embedding_model_name = "cl-nagoya/ruri-v3-310m"
                mock_settings.ingest_device = "cpu"

                from models.embeddings import get_embeddings
                q = get_embeddings(role="query")
                i = get_embeddings(role="ingest")

        assert q is not i

    def test_no_args_defaults_to_query(self, reset_embeddings_cache):
        """get_embeddings() with no arguments returns same object as get_embeddings('query')."""
        mock_cls = MagicMock()
        shared_instance = MagicMock()
        mock_cls.return_value = shared_instance

        with patch("models.embeddings.HuggingFaceEmbeddings", mock_cls):
            with patch("models.embeddings.settings") as mock_settings:
                mock_settings.embedding_model_name = "cl-nagoya/ruri-v3-310m"
                mock_settings.ingest_device = "cpu"

                from models.embeddings import get_embeddings
                default_result = get_embeddings()
                query_result = get_embeddings(role="query")

        assert default_result is query_result
        # __init__ called only once (both hits the same cache)
        assert mock_cls.call_count == 1


# ==========================================================================
# C. compute_faithfulness / compute_context_precision
# ==========================================================================

# Pre-load evaluation.metrics so that patch() can resolve "evaluation.metrics.*".
# This import will succeed (module exists) even though the target functions are
# not yet implemented — those will raise AttributeError inside the tests.
try:
    import evaluation.metrics as _eval_metrics_preload  # noqa: F401
except Exception:
    pass  # If import fails entirely, individual tests will report ImportError

# --- Helpers ---

def _make_ragas_mock(score: float):
    """Return an AsyncMock metric that resolves to ``score``."""
    mock_metric = MagicMock()
    mock_metric.single_turn_ascore = AsyncMock(return_value=score)
    return mock_metric


FAITHFULNESS_THRESHOLD_DEFAULT = 0.7
CONTEXT_PRECISION_THRESHOLD_DEFAULT = 0.7


# ==========================================================================
# C-1. compute_faithfulness
# ==========================================================================

class TestComputeFaithfulnessNormal:
    """Normal cases for compute_faithfulness."""

    @pytest.mark.asyncio
    async def test_pass_when_score_above_threshold(self):
        """score=0.9, threshold=0.7 -> verdict='PASS'."""
        mock_metric = _make_ragas_mock(0.9)
        with patch("evaluation.metrics.Faithfulness", return_value=mock_metric, create=True):
            with patch("evaluation.metrics.settings") as mock_settings:
                mock_settings.ragas_faithfulness_threshold = 0.7
                mock_settings.ragas_response_relevancy_threshold = 0.7

                from evaluation.metrics import compute_faithfulness
                result = await compute_faithfulness(
                    question="What is AI?",
                    context="AI stands for Artificial Intelligence.",
                    answer="AI is Artificial Intelligence.",
                )

        assert result["verdict"] == "PASS"
        assert result["score"] == pytest.approx(0.9, abs=1e-3)
        assert result["threshold"] == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_fail_when_score_below_threshold(self):
        """score=0.5, threshold=0.7 -> verdict='FAIL'."""
        mock_metric = _make_ragas_mock(0.5)
        with patch("evaluation.metrics.Faithfulness", return_value=mock_metric, create=True):
            with patch("evaluation.metrics.settings") as mock_settings:
                mock_settings.ragas_faithfulness_threshold = 0.7

                from evaluation.metrics import compute_faithfulness
                result = await compute_faithfulness(
                    question="What is AI?",
                    context="AI stands for Artificial Intelligence.",
                    answer="AI is Artificial Intelligence.",
                )

        assert result["verdict"] == "FAIL"
        assert result["score"] == pytest.approx(0.5, abs=1e-3)

    @pytest.mark.asyncio
    async def test_pass_when_score_equals_threshold(self):
        """score=0.7 == threshold=0.7 -> verdict='PASS' (boundary: >= passes)."""
        mock_metric = _make_ragas_mock(0.7)
        with patch("evaluation.metrics.Faithfulness", return_value=mock_metric, create=True):
            with patch("evaluation.metrics.settings") as mock_settings:
                mock_settings.ragas_faithfulness_threshold = 0.7

                from evaluation.metrics import compute_faithfulness
                result = await compute_faithfulness(
                    question="Q",
                    context="C",
                    answer="A",
                )

        assert result["verdict"] == "PASS"


class TestComputeFaithfulnessError:
    """Error/fallback cases for compute_faithfulness."""

    @pytest.mark.asyncio
    async def test_ragas_exception_returns_pass_fallback(self):
        """RAGAS raises exception -> score=0.0, verdict='PASS', threshold unchanged."""
        mock_metric = MagicMock()
        mock_metric.single_turn_ascore = AsyncMock(side_effect=RuntimeError("RAGAS error"))
        with patch("evaluation.metrics.Faithfulness", return_value=mock_metric, create=True):
            with patch("evaluation.metrics.settings") as mock_settings:
                mock_settings.ragas_faithfulness_threshold = 0.7

                from evaluation.metrics import compute_faithfulness
                result = await compute_faithfulness(
                    question="Q",
                    context="C",
                    answer="A",
                )

        assert result["score"] == 0.0
        assert result["verdict"] == "PASS"
        assert result["threshold"] == pytest.approx(0.7)


class TestComputeFaithfulnessEdge:
    """Edge cases for compute_faithfulness."""

    @pytest.mark.asyncio
    async def test_empty_context_no_error(self):
        """context='' -> no exception raised (graceful degradation)."""
        mock_metric = _make_ragas_mock(0.8)
        with patch("evaluation.metrics.Faithfulness", return_value=mock_metric, create=True):
            with patch("evaluation.metrics.settings") as mock_settings:
                mock_settings.ragas_faithfulness_threshold = 0.7

                from evaluation.metrics import compute_faithfulness
                result = await compute_faithfulness(
                    question="Q",
                    context="",
                    answer="A",
                )

        assert "verdict" in result
        assert "score" in result
        assert "threshold" in result

    @pytest.mark.asyncio
    async def test_empty_answer_no_error(self):
        """answer='' -> no exception raised (graceful degradation)."""
        mock_metric = _make_ragas_mock(0.6)
        with patch("evaluation.metrics.Faithfulness", return_value=mock_metric, create=True):
            with patch("evaluation.metrics.settings") as mock_settings:
                mock_settings.ragas_faithfulness_threshold = 0.7

                from evaluation.metrics import compute_faithfulness
                result = await compute_faithfulness(
                    question="Q",
                    context="Some context.",
                    answer="",
                )

        assert "verdict" in result
        assert "score" in result


class TestComputeFaithfulnessIdempotency:
    """Idempotency: same input twice produces same verdict."""

    @pytest.mark.asyncio
    async def test_same_input_same_verdict(self):
        """Calling compute_faithfulness twice with identical inputs -> same verdict."""
        mock_metric = _make_ragas_mock(0.85)
        with patch("evaluation.metrics.Faithfulness", return_value=mock_metric, create=True):
            with patch("evaluation.metrics.settings") as mock_settings:
                mock_settings.ragas_faithfulness_threshold = 0.7

                from evaluation.metrics import compute_faithfulness
                r1 = await compute_faithfulness(
                    question="Idempotency test",
                    context="Context text.",
                    answer="Answer text.",
                )
                r2 = await compute_faithfulness(
                    question="Idempotency test",
                    context="Context text.",
                    answer="Answer text.",
                )

        assert r1["verdict"] == r2["verdict"]


class TestComputeFaithfulnessSecurity:
    """Security: question/context/answer must not appear in logs; score/threshold may."""

    @pytest.mark.asyncio
    async def test_pii_not_logged(self, caplog):
        """question, context, answer text must not appear in log output."""
        secret_question = "my_secret_question_xyz"
        secret_context = "my_secret_context_abc"
        secret_answer = "my_secret_answer_def"

        mock_metric = _make_ragas_mock(0.9)
        with patch("evaluation.metrics.Faithfulness", return_value=mock_metric, create=True):
            with patch("evaluation.metrics.settings") as mock_settings:
                mock_settings.ragas_faithfulness_threshold = 0.7

                from evaluation.metrics import compute_faithfulness
                with caplog.at_level(logging.DEBUG):
                    await compute_faithfulness(
                        question=secret_question,
                        context=secret_context,
                        answer=secret_answer,
                    )

        log_text = caplog.text
        assert secret_question not in log_text, "Question must not appear in logs"
        assert secret_context not in log_text, "Context must not appear in logs"
        assert secret_answer not in log_text, "Answer must not appear in logs"


# ==========================================================================
# C-2. compute_context_precision
# ==========================================================================

class TestComputeContextPrecisionNormal:
    """Normal cases for compute_context_precision."""

    @pytest.mark.asyncio
    async def test_pass_when_score_above_threshold(self):
        """score=0.9, threshold=0.7 -> verdict='PASS'."""
        mock_metric = _make_ragas_mock(0.9)
        with patch("evaluation.metrics.ContextPrecision", return_value=mock_metric, create=True):
            with patch("evaluation.metrics.settings") as mock_settings:
                mock_settings.ragas_context_precision_threshold = 0.7

                from evaluation.metrics import compute_context_precision
                result = await compute_context_precision(
                    question="What is AI?",
                    context="AI stands for Artificial Intelligence.",
                    answer="AI is Artificial Intelligence.",
                    reference="AI stands for Artificial Intelligence.",
                )

        assert result["verdict"] == "PASS"
        assert result["score"] == pytest.approx(0.9, abs=1e-3)
        assert result["threshold"] == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_fail_when_score_below_threshold(self):
        """score=0.5, threshold=0.7 -> verdict='FAIL'."""
        mock_metric = _make_ragas_mock(0.5)
        with patch("evaluation.metrics.ContextPrecision", return_value=mock_metric, create=True):
            with patch("evaluation.metrics.settings") as mock_settings:
                mock_settings.ragas_context_precision_threshold = 0.7

                from evaluation.metrics import compute_context_precision
                result = await compute_context_precision(
                    question="What is AI?",
                    context="AI stands for Artificial Intelligence.",
                    answer="AI is Artificial Intelligence.",
                    reference="AI stands for Artificial Intelligence.",
                )

        assert result["verdict"] == "FAIL"
        assert result["score"] == pytest.approx(0.5, abs=1e-3)

    @pytest.mark.asyncio
    async def test_pass_when_score_equals_threshold(self):
        """score=0.7 == threshold=0.7 -> verdict='PASS' (boundary: >= passes)."""
        mock_metric = _make_ragas_mock(0.7)
        with patch("evaluation.metrics.ContextPrecision", return_value=mock_metric, create=True):
            with patch("evaluation.metrics.settings") as mock_settings:
                mock_settings.ragas_context_precision_threshold = 0.7

                from evaluation.metrics import compute_context_precision
                result = await compute_context_precision(
                    question="Q",
                    context="C",
                    answer="A",
                    reference="expected answer",
                )

        assert result["verdict"] == "PASS"


class TestComputeContextPrecisionError:
    """Error/fallback cases for compute_context_precision."""

    @pytest.mark.asyncio
    async def test_ragas_exception_returns_pass_fallback(self):
        """RAGAS raises exception -> score=0.0, verdict='PASS', threshold unchanged."""
        mock_metric = MagicMock()
        mock_metric.single_turn_ascore = AsyncMock(side_effect=RuntimeError("RAGAS error"))
        with patch("evaluation.metrics.ContextPrecision", return_value=mock_metric, create=True):
            with patch("evaluation.metrics.settings") as mock_settings:
                mock_settings.ragas_context_precision_threshold = 0.7

                from evaluation.metrics import compute_context_precision
                result = await compute_context_precision(
                    question="Q",
                    context="C",
                    answer="A",
                    reference="expected answer",
                )

        assert result["score"] == 0.0
        assert result["verdict"] == "PASS"
        assert result["threshold"] == pytest.approx(0.7)


class TestComputeContextPrecisionEdge:
    """Edge cases for compute_context_precision."""

    @pytest.mark.asyncio
    async def test_empty_context_no_error(self):
        """context='' -> no exception raised (graceful degradation)."""
        mock_metric = _make_ragas_mock(0.8)
        with patch("evaluation.metrics.ContextPrecision", return_value=mock_metric, create=True):
            with patch("evaluation.metrics.settings") as mock_settings:
                mock_settings.ragas_context_precision_threshold = 0.7

                from evaluation.metrics import compute_context_precision
                result = await compute_context_precision(
                    question="Q",
                    context="",
                    answer="A",
                )

        assert "verdict" in result
        assert "score" in result
        assert "threshold" in result

    @pytest.mark.asyncio
    async def test_empty_answer_no_error(self):
        """answer='' -> no exception raised (graceful degradation)."""
        mock_metric = _make_ragas_mock(0.6)
        with patch("evaluation.metrics.ContextPrecision", return_value=mock_metric, create=True):
            with patch("evaluation.metrics.settings") as mock_settings:
                mock_settings.ragas_context_precision_threshold = 0.7

                from evaluation.metrics import compute_context_precision
                result = await compute_context_precision(
                    question="Q",
                    context="Some context.",
                    answer="",
                )

        assert "verdict" in result
        assert "score" in result


class TestComputeContextPrecisionIdempotency:
    """Idempotency: same input twice produces same verdict."""

    @pytest.mark.asyncio
    async def test_same_input_same_verdict(self):
        """Calling compute_context_precision twice with identical inputs -> same verdict."""
        mock_metric = _make_ragas_mock(0.85)
        with patch("evaluation.metrics.ContextPrecision", return_value=mock_metric, create=True):
            with patch("evaluation.metrics.settings") as mock_settings:
                mock_settings.ragas_context_precision_threshold = 0.7

                from evaluation.metrics import compute_context_precision
                r1 = await compute_context_precision(
                    question="Idempotency test",
                    context="Context text.",
                    answer="Answer text.",
                )
                r2 = await compute_context_precision(
                    question="Idempotency test",
                    context="Context text.",
                    answer="Answer text.",
                )

        assert r1["verdict"] == r2["verdict"]


class TestComputeContextPrecisionSecurity:
    """Security: question/context/answer must not appear in logs; score/threshold may."""

    @pytest.mark.asyncio
    async def test_pii_not_logged(self, caplog):
        """question, context, answer text must not appear in log output."""
        secret_question = "my_secret_question_xyz"
        secret_context = "my_secret_context_abc"
        secret_answer = "my_secret_answer_def"

        mock_metric = _make_ragas_mock(0.9)
        with patch("evaluation.metrics.ContextPrecision", return_value=mock_metric, create=True):
            with patch("evaluation.metrics.settings") as mock_settings:
                mock_settings.ragas_context_precision_threshold = 0.7

                from evaluation.metrics import compute_context_precision
                with caplog.at_level(logging.DEBUG):
                    await compute_context_precision(
                        question=secret_question,
                        context=secret_context,
                        answer=secret_answer,
                    )

        log_text = caplog.text
        assert secret_question not in log_text, "Question must not appear in logs"
        assert secret_context not in log_text, "Context must not appear in logs"
        assert secret_answer not in log_text, "Answer must not appear in logs"
