"""
Unit tests for Phase 4F: OpenSearch index/pipeline management (app/models/opensearch.py).

Tests cover:
  - Normal: index creation on missing, skip on existing; pipeline creation on missing, skip on existing
  - Edge: uppercase datasource normalised to lowercase, knn_vector dimension matches adapter
  - Idempotency: get_or_create_index called twice -> indices.create called only once
  - Security: datasource starting with '_' still gets prefix (avoids reserved-name collision)

OpenSearch client is stubbed with MagicMock. Source does not exist yet -> ImportError -> pytest.skip.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import pytest
from unittest.mock import MagicMock, call, patch

# ---------------------------------------------------------------------------
# Conditional import — skip entire module when source is not yet implemented
# ---------------------------------------------------------------------------

try:
    from models.opensearch import get_or_create_index, get_or_create_search_pipeline, get_os_client
    _IMPORT_OK = True
except ImportError:
    _IMPORT_OK = False


def _skip_if_missing():
    if not _IMPORT_OK:
        pytest.skip("models.opensearch not implemented yet")


# ---------------------------------------------------------------------------
# Helpers: build a stub OpenSearch client
# ---------------------------------------------------------------------------


def _make_os_client(index_exists: bool = False, pipeline_404: bool = True):
    """Return a MagicMock mimicking opensearchpy.OpenSearch.

    Args:
        index_exists: return value of indices.exists()
        pipeline_404: if True, http.get raises Exception("404")
    """
    client = MagicMock()
    client.indices.exists.return_value = index_exists
    client.indices.create.return_value = {"acknowledged": True}

    if pipeline_404:
        client.http.get.side_effect = Exception("404")
    else:
        client.http.get.return_value = {"status": 200}
    client.http.put.return_value = {"acknowledged": True}

    return client


# ===========================================================================
# Normal cases
# ===========================================================================


class TestNormalCasesIndex:
    def test_index_created_when_not_exists(self):
        """Index does not exist -> indices.create is called, index name is returned."""
        _skip_if_missing()
        mock_client = _make_os_client(index_exists=False)

        with (
            patch("models.opensearch.get_os_client", return_value=mock_client),
            patch("models.opensearch.settings") as mock_settings,
        ):
            mock_settings.os_index_prefix = "ls_"
            mock_settings.embedding_model_name = "BAAI/bge-m3"

            index_name = get_or_create_index("my-datasource")

        assert mock_client.indices.create.called, "indices.create should be called"
        assert index_name == "ls_my-datasource"

    def test_index_not_created_when_already_exists(self):
        """Index already exists -> indices.create is NOT called."""
        _skip_if_missing()
        mock_client = _make_os_client(index_exists=True)

        with (
            patch("models.opensearch.get_os_client", return_value=mock_client),
            patch("models.opensearch.settings") as mock_settings,
        ):
            mock_settings.os_index_prefix = "ls_"
            mock_settings.embedding_model_name = "BAAI/bge-m3"

            index_name = get_or_create_index("existing-ds")

        assert not mock_client.indices.create.called, "indices.create must not be called for existing index"
        assert index_name == "ls_existing-ds"

    def test_index_name_includes_prefix(self):
        """Returned index name has OS_INDEX_PREFIX prepended."""
        _skip_if_missing()
        mock_client = _make_os_client(index_exists=False)

        with (
            patch("models.opensearch.get_os_client", return_value=mock_client),
            patch("models.opensearch.settings") as mock_settings,
        ):
            mock_settings.os_index_prefix = "ls_"
            mock_settings.embedding_model_name = "BAAI/bge-m3"

            index_name = get_or_create_index("docs")

        assert index_name.startswith("ls_"), f"Expected 'ls_' prefix, got: {index_name!r}"


class TestNormalCasesPipeline:
    def test_pipeline_created_when_not_exists(self):
        """Pipeline does not exist (http.get raises 404) -> http.put is called."""
        _skip_if_missing()
        mock_client = _make_os_client(pipeline_404=True)

        with (
            patch("models.opensearch.get_os_client", return_value=mock_client),
            patch("models.opensearch.settings") as mock_settings,
        ):
            mock_settings.hybrid_pipeline_name = "hybrid-bm25-knn"
            mock_settings.embedding_model_name = "BAAI/bge-m3"

            pipeline_name = get_or_create_search_pipeline()

        assert mock_client.http.put.called, "http.put should be called when pipeline is missing"
        assert pipeline_name == "hybrid-bm25-knn"

    def test_pipeline_not_created_when_already_exists(self):
        """Pipeline already exists (http.get returns 200) -> http.put is NOT called."""
        _skip_if_missing()
        mock_client = _make_os_client(pipeline_404=False)

        with (
            patch("models.opensearch.get_os_client", return_value=mock_client),
            patch("models.opensearch.settings") as mock_settings,
        ):
            mock_settings.hybrid_pipeline_name = "hybrid-bm25-knn"
            mock_settings.embedding_model_name = "BAAI/bge-m3"

            pipeline_name = get_or_create_search_pipeline()

        assert not mock_client.http.put.called, "http.put must not be called for existing pipeline"
        assert pipeline_name == "hybrid-bm25-knn"


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_uppercase_datasource_normalised_to_lowercase(self):
        """datasource='MyDatasource' -> index name contains lowercase only."""
        _skip_if_missing()
        mock_client = _make_os_client(index_exists=False)

        with (
            patch("models.opensearch.get_os_client", return_value=mock_client),
            patch("models.opensearch.settings") as mock_settings,
        ):
            mock_settings.os_index_prefix = "ls_"
            mock_settings.embedding_model_name = "BAAI/bge-m3"

            index_name = get_or_create_index("MyDatasource")

        assert index_name == index_name.lower(), f"Index name should be lowercase, got: {index_name!r}"
        assert "mydatasource" in index_name

    def test_knn_vector_dimension_matches_bge_m3(self):
        """knn_vector dimension in index mapping == 1024 for bge-m3."""
        _skip_if_missing()
        mock_client = _make_os_client(index_exists=False)
        captured_body = {}

        def capture_create(index, body):
            captured_body.update(body)
            return {"acknowledged": True}

        mock_client.indices.create.side_effect = capture_create

        with (
            patch("models.opensearch.get_os_client", return_value=mock_client),
            patch("models.opensearch.settings") as mock_settings,
        ):
            mock_settings.os_index_prefix = "ls_"
            mock_settings.embedding_model_name = "BAAI/bge-m3"

            get_or_create_index("dim-test")

        # Find the knn_vector field dimension anywhere in the mapping body
        body_str = str(captured_body)
        # dimension for bge-m3 should be 1024
        assert "1024" in body_str, (
            f"Expected knn_vector dimension 1024 for bge-m3 in mapping body, got: {body_str}"
        )


# ===========================================================================
# Idempotency
# ===========================================================================


class TestIdempotency:
    def test_get_or_create_index_twice_creates_only_once(self):
        """get_or_create_index called twice for the same datasource.

        First call: index does not exist -> create.
        Second call: index exists -> skip.
        Overall: indices.create call_count == 1.
        """
        _skip_if_missing()
        mock_client = _make_os_client(index_exists=False)

        # After the first create, simulate that the index now exists
        create_call_count = {"n": 0}
        original_exists = mock_client.indices.exists.return_value

        def exists_side_effect(index_name):
            return create_call_count["n"] > 0

        def create_side_effect(index, body):
            create_call_count["n"] += 1
            return {"acknowledged": True}

        mock_client.indices.exists.side_effect = exists_side_effect
        mock_client.indices.create.side_effect = create_side_effect

        with (
            patch("models.opensearch.get_os_client", return_value=mock_client),
            patch("models.opensearch.settings") as mock_settings,
        ):
            mock_settings.os_index_prefix = "ls_"
            mock_settings.embedding_model_name = "BAAI/bge-m3"

            get_or_create_index("idempotent-ds")
            get_or_create_index("idempotent-ds")

        assert create_call_count["n"] == 1, (
            f"indices.create should be called exactly once, but was called {create_call_count['n']} times"
        )


# ===========================================================================
# Security cases
# ===========================================================================


class TestSecurityCases:
    def test_datasource_leading_underscore_gets_prefix(self):
        """datasource='_secret' -> index name is 'ls__secret' (prefix prevents bare reserved name)."""
        _skip_if_missing()
        mock_client = _make_os_client(index_exists=False)

        with (
            patch("models.opensearch.get_os_client", return_value=mock_client),
            patch("models.opensearch.settings") as mock_settings,
        ):
            mock_settings.os_index_prefix = "ls_"
            mock_settings.embedding_model_name = "BAAI/bge-m3"

            index_name = get_or_create_index("_secret")

        # The prefix must be present; the raw '_secret' must not appear without it
        assert index_name.startswith("ls_"), (
            f"Expected 'ls_' prefix even for '_secret' datasource, got: {index_name!r}"
        )
        assert not index_name.startswith("_"), (
            "Index name must not start with bare '_' — OS reserved namespace"
        )
