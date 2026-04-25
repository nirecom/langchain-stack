"""
Unit tests for Phase 4F: hybrid search query builders (app/rag/retriever.py additions).

Tests cover:
  - Normal: _build_dense, _build_header_dense, _build_hybrid, _build_hybrid_header return
    correct OpenSearch query body structures
  - Edge: k=1 -> effective knn k >= 10 (min_k guarantee); all return dict
  - Idempotency: same args twice -> identical dict (deepcopy comparison)

Source functions (_build_dense etc.) do not exist yet -> ImportError -> pytest.skip.
Tests will activate automatically once retriever.py is updated.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import copy
import pytest

# ---------------------------------------------------------------------------
# Conditional import — skip entire module when source functions are not yet added
# ---------------------------------------------------------------------------

try:
    from rag.retriever import (
        _build_dense,
        _build_header_dense,
        _build_hybrid,
        _build_hybrid_header,
    )
    _IMPORT_OK = True
except ImportError:
    _IMPORT_OK = False


def _skip_if_missing():
    if not _IMPORT_OK:
        pytest.skip("_build_dense/_build_hybrid not implemented in rag.retriever yet")


# ---------------------------------------------------------------------------
# Shared test vector
# ---------------------------------------------------------------------------

_VECTOR = [0.1] * 1024   # 1024-dim for bge-m3
_TEXT = "sample query"
_K = 5


# ===========================================================================
# Normal cases
# ===========================================================================


class TestNormalBuildDense:
    def test_returns_dict(self):
        """_build_dense returns a dict."""
        _skip_if_missing()
        result = _build_dense(_VECTOR, _TEXT, _K)
        assert isinstance(result, dict)

    def test_knn_key_present(self):
        """_build_dense result contains 'knn' key at top level or nested."""
        _skip_if_missing()
        result = _build_dense(_VECTOR, _TEXT, _K)
        body_str = str(result)
        assert "knn" in body_str, f"Expected 'knn' in query body, got: {body_str}"

    def test_vector_embedded_in_query(self):
        """_build_dense embeds the query vector in the body."""
        _skip_if_missing()
        result = _build_dense(_VECTOR, _TEXT, _K)
        body_str = str(result)
        # Check that at least part of our vector is referenced
        assert "0.1" in body_str or str(_VECTOR[0]) in body_str

    def test_k_value_present(self):
        """_build_dense includes k value >= requested k."""
        _skip_if_missing()
        result = _build_dense(_VECTOR, _TEXT, _K)

        def _find_k(obj, depth=0):
            """Recursively find any integer value >= _K in the dict."""
            if depth > 10:
                return False
            if isinstance(obj, dict):
                for v in obj.values():
                    if _find_k(v, depth + 1):
                        return True
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    if _find_k(item, depth + 1):
                        return True
            elif isinstance(obj, int) and obj >= _K:
                return True
            return False

        assert _find_k(result), f"Expected k>={_K} in query body: {result}"


class TestNormalBuildHeaderDense:
    def test_returns_dict(self):
        """_build_header_dense returns a dict."""
        _skip_if_missing()
        result = _build_header_dense(_VECTOR, _TEXT, _K)
        assert isinstance(result, dict)

    def test_knn_key_present(self):
        """_build_header_dense contains knn."""
        _skip_if_missing()
        result = _build_header_dense(_VECTOR, _TEXT, _K)
        body_str = str(result)
        assert "knn" in body_str

    def test_should_match_on_title(self):
        """_build_header_dense has 'title' boost in should/match."""
        _skip_if_missing()
        result = _build_header_dense(_VECTOR, _TEXT, _K)
        body_str = str(result)
        assert "title" in body_str, f"Expected 'title' match boost, got: {body_str}"

    def test_should_match_on_file_name(self):
        """_build_header_dense has 'file_name' boost in should/match."""
        _skip_if_missing()
        result = _build_header_dense(_VECTOR, _TEXT, _K)
        body_str = str(result)
        assert "file_name" in body_str, f"Expected 'file_name' match boost, got: {body_str}"

    def test_bool_or_should_structure(self):
        """_build_header_dense uses bool/should or equivalent structure for boosting."""
        _skip_if_missing()
        result = _build_header_dense(_VECTOR, _TEXT, _K)
        body_str = str(result)
        # At least one of these structural hints should be present
        assert any(kw in body_str for kw in ("should", "bool", "boost")), (
            f"Expected bool/should/boost structure for header boosting, got: {body_str}"
        )


class TestNormalBuildHybrid:
    def test_returns_dict(self):
        """_build_hybrid returns a dict."""
        _skip_if_missing()
        result = _build_hybrid(_VECTOR, _TEXT, _K)
        assert isinstance(result, dict)

    def test_search_pipeline_referenced(self):
        """_build_hybrid references a search pipeline (hybrid-bm25-knn or similar)."""
        _skip_if_missing()
        result = _build_hybrid(_VECTOR, _TEXT, _K)
        body_str = str(result)
        # The pipeline name or a pipeline-related key should appear
        assert any(kw in body_str for kw in ("pipeline", "hybrid", "search_pipeline")), (
            f"Expected pipeline reference in hybrid query body, got: {body_str}"
        )

    def test_multi_match_present(self):
        """_build_hybrid includes multi_match (text BM25 side)."""
        _skip_if_missing()
        result = _build_hybrid(_VECTOR, _TEXT, _K)
        body_str = str(result)
        assert "multi_match" in body_str or "match" in body_str, (
            f"Expected multi_match in hybrid query, got: {body_str}"
        )

    def test_knn_present(self):
        """_build_hybrid includes knn (dense side)."""
        _skip_if_missing()
        result = _build_hybrid(_VECTOR, _TEXT, _K)
        body_str = str(result)
        assert "knn" in body_str


class TestNormalBuildHybridHeader:
    def test_returns_dict(self):
        """_build_hybrid_header returns a dict."""
        _skip_if_missing()
        result = _build_hybrid_header(_VECTOR, _TEXT, _K)
        assert isinstance(result, dict)

    def test_title_boost_in_multi_match(self):
        """_build_hybrid_header multi_match fields include title^2.5."""
        _skip_if_missing()
        result = _build_hybrid_header(_VECTOR, _TEXT, _K)
        body_str = str(result)
        # title should appear with a boost factor (2.5 or as separate field entry)
        assert "title" in body_str, f"Expected 'title' field, got: {body_str}"
        assert "2.5" in body_str, f"Expected boost factor 2.5 for title, got: {body_str}"

    def test_file_name_boost_in_multi_match(self):
        """_build_hybrid_header multi_match fields include file_name^2.0."""
        _skip_if_missing()
        result = _build_hybrid_header(_VECTOR, _TEXT, _K)
        body_str = str(result)
        assert "file_name" in body_str, f"Expected 'file_name' field, got: {body_str}"
        assert "2.0" in body_str, f"Expected boost factor 2.0 for file_name, got: {body_str}"

    def test_section_path_boost_in_multi_match(self):
        """_build_hybrid_header multi_match fields include section_path^1.5."""
        _skip_if_missing()
        result = _build_hybrid_header(_VECTOR, _TEXT, _K)
        body_str = str(result)
        assert "section_path" in body_str, f"Expected 'section_path' field, got: {body_str}"
        assert "1.5" in body_str, f"Expected boost factor 1.5 for section_path, got: {body_str}"

    def test_knn_present(self):
        """_build_hybrid_header includes knn (dense side)."""
        _skip_if_missing()
        result = _build_hybrid_header(_VECTOR, _TEXT, _K)
        body_str = str(result)
        assert "knn" in body_str


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_build_dense_k1_effective_k_at_least_10(self):
        """_build_dense(k=1) -> effective knn k in body >= 10 (min_k guarantee)."""
        _skip_if_missing()
        result = _build_dense(_VECTOR, _TEXT, k=1)

        def _max_int(obj, depth=0):
            if depth > 10:
                return 0
            if isinstance(obj, dict):
                return max((_max_int(v, depth + 1) for v in obj.values()), default=0)
            elif isinstance(obj, (list, tuple)):
                return max((_max_int(item, depth + 1) for item in obj), default=0)
            elif isinstance(obj, int):
                return obj
            return 0

        max_k = _max_int(result)
        assert max_k >= 10, (
            f"Expected effective k >= 10 when k=1, but max int found in body was {max_k}"
        )

    def test_build_hybrid_k1_effective_k_at_least_10(self):
        """_build_hybrid(k=1) -> effective knn k in body >= 10 (min_k guarantee)."""
        _skip_if_missing()
        result = _build_hybrid(_VECTOR, _TEXT, k=1)

        def _max_int(obj, depth=0):
            if depth > 10:
                return 0
            if isinstance(obj, dict):
                return max((_max_int(v, depth + 1) for v in obj.values()), default=0)
            elif isinstance(obj, (list, tuple)):
                return max((_max_int(item, depth + 1) for item in obj), default=0)
            elif isinstance(obj, int):
                return obj
            return 0

        max_k = _max_int(result)
        assert max_k >= 10, (
            f"Expected effective k >= 10 when k=1, but max int found in body was {max_k}"
        )

    def test_all_builders_return_dict(self):
        """All four builder functions return isinstance(result, dict)."""
        _skip_if_missing()
        builders = [_build_dense, _build_header_dense, _build_hybrid, _build_hybrid_header]
        for fn in builders:
            result = fn(_VECTOR, _TEXT, _K)
            assert isinstance(result, dict), f"{fn.__name__} must return dict, got {type(result)}"


# ===========================================================================
# Idempotency
# ===========================================================================


class TestIdempotency:
    def test_build_dense_same_args_same_result(self):
        """_build_dense called twice with same args -> identical dict."""
        _skip_if_missing()
        result1 = _build_dense(_VECTOR, _TEXT, _K)
        result2 = _build_dense(copy.deepcopy(_VECTOR), _TEXT, _K)
        assert result1 == result2, "_build_dense must be deterministic"

    def test_build_header_dense_same_args_same_result(self):
        """_build_header_dense called twice with same args -> identical dict."""
        _skip_if_missing()
        result1 = _build_header_dense(_VECTOR, _TEXT, _K)
        result2 = _build_header_dense(copy.deepcopy(_VECTOR), _TEXT, _K)
        assert result1 == result2, "_build_header_dense must be deterministic"

    def test_build_hybrid_same_args_same_result(self):
        """_build_hybrid called twice with same args -> identical dict."""
        _skip_if_missing()
        result1 = _build_hybrid(_VECTOR, _TEXT, _K)
        result2 = _build_hybrid(copy.deepcopy(_VECTOR), _TEXT, _K)
        assert result1 == result2, "_build_hybrid must be deterministic"

    def test_build_hybrid_header_same_args_same_result(self):
        """_build_hybrid_header called twice with same args -> identical dict."""
        _skip_if_missing()
        result1 = _build_hybrid_header(_VECTOR, _TEXT, _K)
        result2 = _build_hybrid_header(copy.deepcopy(_VECTOR), _TEXT, _K)
        assert result1 == result2, "_build_hybrid_header must be deterministic"
