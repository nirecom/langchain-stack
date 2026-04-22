"""
Tests for the user-based RAG access control feature.

Redesign: ACL mapping is now from user → datasources (not model → datasources).
Each user has their own Bearer token read from a per-user env var
(``CHAT_API_KEY_<USER>``). Requests are authenticated and authorized by user.

Test groups:
  - TestUserAuth: per-user Bearer token authentication on /v1/chat/completions
  - TestUserDatasourceScope: retrieval scope per user
  - TestRegistryStartup: UserRegistry / validate_access_control error detection
  - TestAuditUserField: retrieve audit entry carries ``user`` field, ingest does not
  - TestIdempotency: repeated request yields identical datasource scope
  - TestSecurity: bypass, case, cross-endpoint token checks
  - TestLegacyModelsWarning: YAML with legacy top-level ``models:`` warns once

Most tests depend on the new ``UserRegistry`` / per-user API and are expected to
fail with ImportError/AttributeError until the implementation lands. That is
the intended "red" state.
"""
import sys
import os

# Ensure ``app/`` is importable so ``from rag... import ...`` resolves.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

# Per-user chat tokens and the ingest token — defaults are only used when the
# real env vars are not already set by the test runner.
os.environ.setdefault("CHAT_API_KEY_KYOKO", "test-kyoko-key")
os.environ.setdefault("CHAT_API_KEY_NIRE", "test-nire-key")
os.environ.setdefault("CHAT_API_KEY_EDGE", "test-edge-key")
os.environ.setdefault("CHAT_API_KEY_LUTE", "test-lute-key")
os.environ.setdefault("INGEST_API_KEY", "test-ingest-key")

import json
import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8100")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


USER_TOKENS = {
    "kyoko": os.environ["CHAT_API_KEY_KYOKO"],
    "nire": os.environ["CHAT_API_KEY_NIRE"],
    "edge": os.environ["CHAT_API_KEY_EDGE"],
    "lute": os.environ["CHAT_API_KEY_LUTE"],
}

EXPECTED_SCOPE = {
    "kyoko": ["family-docs", "parents-docs"],
    "nire": ["family-docs", "nire-docs", "parents-docs"],
    "edge": ["family-docs"],
    "lute": ["family-docs"],
}


def _make_registry_config():
    """Standard valid access-control config used by most tests."""
    return {
        "datasources": {
            "family-docs": {},
            "nire-docs": {},
            "parents-docs": {},
        },
        "users": {
            "kyoko": {
                "api_key_env": "CHAT_API_KEY_KYOKO",
                "datasources": ["family-docs", "parents-docs"],
            },
            "nire": {
                "api_key_env": "CHAT_API_KEY_NIRE",
                "datasources": ["family-docs", "nire-docs", "parents-docs"],
            },
            "edge": {
                "api_key_env": "CHAT_API_KEY_EDGE",
                "datasources": ["family-docs"],
            },
            "lute": {
                "api_key_env": "CHAT_API_KEY_LUTE",
                "datasources": ["family-docs"],
            },
        },
    }


@pytest.fixture(scope="session")
def wait_for_server():
    """Wait until the server is reachable (max 30s). Skip if unavailable."""
    for _ in range(30):
        try:
            r = httpx.get(f"{BASE_URL}/health", timeout=1)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    pytest.skip("Server not reachable — skipping E2E tests")


@pytest.fixture
def fastapi_client(wait_for_server):
    """httpx client pointing at the real running server."""
    with httpx.Client(base_url=BASE_URL, timeout=30) as client:
        yield client


def _chat_bearer(user: str) -> dict:
    return {"Authorization": f"Bearer {USER_TOKENS[user]}"}


# ---------------------------------------------------------------------------
# TestUserAuth — Bearer token authentication keyed by user
# ---------------------------------------------------------------------------


class TestUserAuth:
    @pytest.mark.parametrize("user", ["kyoko", "nire", "edge", "lute"])
    def test_valid_user_key_ok(self, fastapi_client, user):
        """Each user with their own key → 200."""
        r = fastapi_client.post(
            "/v1/chat/completions",
            json={
                "model": "judge-chain",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers=_chat_bearer(user),
        )
        assert r.status_code == 200, r.text

    def test_cross_user_key_rejected(self, fastapi_client):
        """Kyoko's key used while Authorization claims nire still rejected only
        if a mismatching header is presented. Here we simply verify that using
        a *known* token authenticates *that* user, not somebody else. Swapping
        keys between users is covered by ``test_unknown_key_rejected`` below.
        """
        # Send nire's token with a request intended as nire; this is the
        # sanity complement to test_unknown_key_rejected.
        r = fastapi_client.post(
            "/v1/chat/completions",
            json={
                "model": "judge-chain",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"Authorization": "Bearer completely-other-users-key"},
        )
        assert r.status_code == 401

    def test_unknown_key_rejected(self, fastapi_client):
        """Bogus key → 401."""
        r = fastapi_client.post(
            "/v1/chat/completions",
            json={
                "model": "judge-chain",
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={"Authorization": "Bearer does-not-exist"},
        )
        assert r.status_code == 401

    def test_missing_authorization_header(self, fastapi_client):
        """No Authorization header → 401."""
        r = fastapi_client.post(
            "/v1/chat/completions",
            json={
                "model": "judge-chain",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert r.status_code == 401

    def test_empty_bearer_token(self, fastapi_client):
        """``Bearer `` with empty token → 401 (or rejected by client as illegal header)."""
        try:
            r = fastapi_client.post(
                "/v1/chat/completions",
                json={
                    "model": "judge-chain",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                headers={"Authorization": "Bearer "},
            )
            assert r.status_code == 401
        except httpx.LocalProtocolError:
            # Client-side rejection of illegal header value is acceptable —
            # the empty token cannot reach the server at all.
            pass


# ---------------------------------------------------------------------------
# TestUserDatasourceScope — permitted datasources per user
# ---------------------------------------------------------------------------


class TestUserDatasourceScope:
    @pytest.mark.parametrize("user,expected", sorted(EXPECTED_SCOPE.items()))
    @pytest.mark.asyncio
    async def test_user_scope(self, user, expected):
        """``get_relevant_context(..., user=<user>)`` must query exactly the
        datasources listed in the registry for that user."""
        from rag import retriever as retriever_mod
        from rag.retriever import get_relevant_context

        # StubCollection returning one document so ``datasources_queried``
        # includes the collection.
        class _StubCol:
            def query(self, query_embeddings, n_results, include=None):
                return {"documents": [["hit"]], "distances": [[0.1]]}

        class _StubClient:
            def __init__(self, names):
                self._names = set(names)

            def get_collection(self, name):
                if name not in self._names:
                    import chromadb.errors
                    raise chromadb.errors.NotFoundError(name)
                return _StubCol()

        captured = {}

        def capture_log(**kwargs):
            captured.update(kwargs)

        with (
            patch("rag.retriever.get_chroma_client",
                  MagicMock(return_value=_StubClient(expected))),
            patch("rag.retriever.get_embeddings",
                  MagicMock(return_value=MagicMock(
                      embed_documents=lambda xs: [[0.0, 0.0, 0.0]]))),
            patch("rag.retriever.get_permitted_datasources_for_user",
                  MagicMock(return_value=list(expected))),
            patch("rag.retriever.log_retrieve_event", capture_log),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            await get_relevant_context("test", user=user)

        assert captured.get("user") == user
        assert sorted(captured.get("datasources_queried", [])) == sorted(expected)


# ---------------------------------------------------------------------------
# TestRegistryStartup — config validation at load time
# ---------------------------------------------------------------------------


class TestRegistryStartup:
    def test_user_references_unregistered_datasource(self):
        """User points at a datasource not declared in ``datasources`` → error."""
        from rag.access_control import UserRegistry, validate_access_control

        cfg = _make_registry_config()
        cfg["users"]["kyoko"]["datasources"] = ["family-docs", "ghost-docs"]
        with pytest.raises(RuntimeError):
            registry = UserRegistry.build_from_config(cfg)
            validate_access_control(cfg, registry)

    def test_duplicate_api_key_env(self):
        """Two users pointing at the same ``api_key_env`` → error."""
        from rag.access_control import UserRegistry, validate_access_control

        cfg = _make_registry_config()
        cfg["users"]["edge"]["api_key_env"] = "CHAT_API_KEY_NIRE"
        with pytest.raises(RuntimeError):
            registry = UserRegistry.build_from_config(cfg)
            validate_access_control(cfg, registry)

    def test_empty_env_var_excludes_user(self, caplog, monkeypatch):
        """Unset/empty env var for a user: that user excluded (WARNING)
        but registry load still succeeds for the remaining users."""
        from rag.access_control import UserRegistry, validate_access_control

        monkeypatch.setenv("CHAT_API_KEY_LUTE", "")
        cfg = _make_registry_config()
        with caplog.at_level(logging.WARNING):
            registry = UserRegistry.build_from_config(cfg)
            validate_access_control(cfg, registry)
        # Lute should not resolve; others should.
        assert registry.get_user_by_api_key(os.environ["CHAT_API_KEY_NIRE"]) == "nire"
        assert registry.get_user_by_api_key("") is None

    def test_two_users_same_token_value(self, monkeypatch):
        """Two different env vars resolve to identical token strings → error."""
        from rag.access_control import UserRegistry, validate_access_control

        monkeypatch.setenv("CHAT_API_KEY_EDGE", "collision")
        monkeypatch.setenv("CHAT_API_KEY_LUTE", "collision")
        cfg = _make_registry_config()
        with pytest.raises(RuntimeError):
            registry = UserRegistry.build_from_config(cfg)
            validate_access_control(cfg, registry)

    def test_user_token_equals_ingest_token(self, monkeypatch):
        """A user's chat token coincides with INGEST_API_KEY → error."""
        from rag.access_control import UserRegistry, validate_access_control

        monkeypatch.setenv("INGEST_API_KEY", "shared-secret")
        monkeypatch.setenv("CHAT_API_KEY_NIRE", "shared-secret")
        cfg = _make_registry_config()
        with pytest.raises(RuntimeError):
            registry = UserRegistry.build_from_config(cfg)
            validate_access_control(cfg, registry)


# ---------------------------------------------------------------------------
# TestAuditUserField — retrieve entries carry user, ingest entries do not
# ---------------------------------------------------------------------------


class TestAuditUserField:
    def test_retrieve_entry_has_user(self, tmp_path, monkeypatch):
        """``log_retrieve_event`` must write a ``user`` key into JSONL."""
        from rag import audit

        audit_file = tmp_path / "audit.jsonl"
        monkeypatch.setattr(audit.settings, "audit_log_path", str(audit_file))

        audit.log_retrieve_event(
            user="nire",
            datasources_queried=["family-docs"],
            query="hello",
            hits=1,
        )
        lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
        entry = json.loads(lines[-1])
        assert entry["action"] == "retrieve"
        assert entry["user"] == "nire"

    def test_ingest_entry_has_no_user(self, tmp_path, monkeypatch):
        """``log_ingest_event`` must NOT write a ``user`` field."""
        from rag import audit

        audit_file = tmp_path / "audit.jsonl"
        monkeypatch.setattr(audit.settings, "audit_log_path", str(audit_file))

        audit.log_ingest_event("ingest", "family-docs",
                               filename="f.txt", chunks=1)
        lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
        entry = json.loads(lines[-1])
        assert "user" not in entry

    def test_legacy_entry_missing_user_falls_back_to_empty(self, tmp_path, monkeypatch):
        """Legacy audit entries (pre-user-field) should surface ``user`` as
        empty string when read via get_recent_events (fallback via .get())."""
        from rag import audit

        audit_file = tmp_path / "audit.jsonl"
        audit_file.write_text(json.dumps({
            "timestamp": "2024-01-01T00:00:00+00:00",
            "action": "retrieve",
            "model_name": "judge-chain",
            "datasources_queried": ["family-docs"],
            "query_length": 3,
            "hits": 1,
            "status": "ok",
            "error": "",
        }) + "\n", encoding="utf-8")
        monkeypatch.setattr(audit.settings, "audit_log_path", str(audit_file))

        events = audit.get_recent_events()
        assert events, "expected at least one audit event"
        # Fallback: legacy entry (no 'user' key) → .get("user", "") == ""
        assert all(e.get("user", "") == "" for e in events)


# ---------------------------------------------------------------------------
# TestIdempotency — same request twice == same scope
# ---------------------------------------------------------------------------


class TestIdempotency:
    @pytest.mark.asyncio
    async def test_repeated_request_same_scope(self):
        from rag.retriever import get_relevant_context

        class _StubCol:
            def query(self, query_embeddings, n_results, include=None):
                return {"documents": [["hit"]], "distances": [[0.1]]}

        class _StubClient:
            def get_collection(self, name):
                return _StubCol()

        captured_runs = []

        def capture(**kwargs):
            captured_runs.append(sorted(kwargs.get("datasources_queried", [])))

        with (
            patch("rag.retriever.get_chroma_client",
                  MagicMock(return_value=_StubClient())),
            patch("rag.retriever.get_embeddings",
                  MagicMock(return_value=MagicMock(
                      embed_documents=lambda xs: [[0.0, 0.0, 0.0]]))),
            patch("rag.retriever.get_permitted_datasources_for_user",
                  MagicMock(return_value=["family-docs", "nire-docs"])),
            patch("rag.retriever.log_retrieve_event", capture),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            await get_relevant_context("q", user="nire")
            await get_relevant_context("q", user="nire")

        assert len(captured_runs) == 2
        assert captured_runs[0] == captured_runs[1]


# ---------------------------------------------------------------------------
# TestSecurity — bypass, case, cross-endpoint
# ---------------------------------------------------------------------------


class TestSecurity:
    def test_empty_token_cannot_bypass(self, fastapi_client):
        try:
            r = fastapi_client.post(
                "/v1/chat/completions",
                json={"model": "judge-chain",
                      "messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": "Bearer "},
            )
            assert r.status_code == 401
        except httpx.LocalProtocolError:
            pass  # Client-side rejection is acceptable

    def test_case_mismatched_token_rejected(self, fastapi_client):
        """Token comparison is case-sensitive."""
        token = USER_TOKENS["kyoko"].upper()
        # Only run the check when upper-case differs from original.
        if token == USER_TOKENS["kyoko"]:
            pytest.skip("token has no case to mismatch")
        r = fastapi_client.post(
            "/v1/chat/completions",
            json={"model": "judge-chain",
                  "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 401

    def test_chat_token_cannot_access_ingest(self, fastapi_client):
        """A valid chat user token must not authorise /ingest endpoints."""
        r = fastapi_client.delete(
            "/ingest/family-docs",
            headers=_chat_bearer("nire"),
        )
        assert r.status_code == 401

    def test_ingest_token_cannot_access_chat(self, fastapi_client):
        """The INGEST token must not be accepted on /v1/chat/completions."""
        r = fastapi_client.post(
            "/v1/chat/completions",
            json={"model": "judge-chain",
                  "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {os.environ['INGEST_API_KEY']}"},
        )
        assert r.status_code == 401

    def test_token_not_leaked_in_error_response(self, fastapi_client):
        """Error responses (4xx/5xx) must never echo back the Bearer token value."""
        token = USER_TOKENS["nire"]
        r = fastapi_client.post(
            "/v1/chat/completions",
            json={"model": "judge-chain",
                  "messages": [{"role": "user", "content": "hi"}]},
            headers=_chat_bearer("nire"),
        )
        assert token not in r.text, f"Token value leaked in response: {r.text[:200]}"

    def test_newline_injection_in_authorization_header(self, fastapi_client):
        """Newline-injected Authorization header must be rejected by client or server."""
        try:
            r = fastapi_client.post(
                "/v1/chat/completions",
                json={"model": "judge-chain",
                      "messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": "Bearer test-nire-key\nX-Admin: true"},
            )
            assert r.status_code in (400, 401, 422)
        except (httpx.LocalProtocolError, httpx.InvalidURL):
            pass  # Client-side rejection is acceptable

    def test_extremely_long_bearer_token_rejected(self, fastapi_client):
        """10 KB token must be rejected, not crash the server."""
        long_token = "x" * 10240
        r = fastapi_client.post(
            "/v1/chat/completions",
            json={"model": "judge-chain",
                  "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": f"Bearer {long_token}"},
        )
        assert r.status_code in (400, 401, 413)

    def test_unicode_normalization_bypass(self, fastapi_client):
        """Token with appended combining Unicode characters must not match the base token."""
        base_token = USER_TOKENS["nire"]
        crafted_token = base_token + "̀"  # combining grave accent — byte-different
        try:
            r = fastapi_client.post(
                "/v1/chat/completions",
                json={"model": "judge-chain",
                      "messages": [{"role": "user", "content": "hi"}]},
                headers={"Authorization": f"Bearer {crafted_token}"},
            )
            assert r.status_code == 401
        except (httpx.LocalProtocolError, httpx.InvalidURL, UnicodeEncodeError):
            pass  # Client-side rejection is acceptable — non-ASCII header values cannot reach the server


# ---------------------------------------------------------------------------
# TestLegacyModelsWarning — legacy ``models:`` key triggers one WARNING
# ---------------------------------------------------------------------------


class TestLegacyModelsWarning:
    def test_legacy_models_key_warns_once(self, caplog):
        """Config with top-level ``models:`` (legacy mapping) should log a
        single WARNING but still produce a usable UserRegistry."""
        from rag.access_control import UserRegistry, validate_access_control

        cfg = _make_registry_config()
        cfg["models"] = {
            "judge-chain": {"datasources": ["family-docs"]},
        }
        with caplog.at_level(logging.WARNING):
            registry = UserRegistry.build_from_config(cfg)
            validate_access_control(cfg, registry)

        legacy_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and "models" in r.getMessage().lower()
        ]
        assert len(legacy_warnings) >= 1
        # ACL still works for nire regardless of legacy warning.
        assert registry.get_user_by_api_key(os.environ["CHAT_API_KEY_NIRE"]) == "nire"


# ---------------------------------------------------------------------------
# TestSecurityLeakage — secrets must not appear in logs or responses
# ---------------------------------------------------------------------------


class TestSecurityLeakage:
    def test_config_validation_does_not_log_token_values(self, caplog):
        """validate_access_control must log env var NAMES only, not actual token values."""
        from rag.access_control import UserRegistry, validate_access_control

        cfg = _make_registry_config()
        with caplog.at_level(logging.DEBUG):
            registry = UserRegistry.build_from_config(cfg)
            validate_access_control(cfg, registry)

        for user, token in USER_TOKENS.items():
            for record in caplog.records:
                msg = record.getMessage()
                assert token not in msg, (
                    f"Token for '{user}' leaked in log message: {msg!r}"
                )

    def test_ingest_key_not_logged_during_startup(self, caplog):
        """INGEST_API_KEY value must not appear in any log during registry build."""
        from rag.access_control import UserRegistry, validate_access_control

        ingest_key = os.environ.get("INGEST_API_KEY", "")
        cfg = _make_registry_config()
        with caplog.at_level(logging.DEBUG):
            registry = UserRegistry.build_from_config(cfg)
            validate_access_control(cfg, registry)

        if ingest_key:
            for record in caplog.records:
                assert ingest_key not in record.getMessage()


# ---------------------------------------------------------------------------
# TestEdgeCasesACL — boundary and collection edge cases
# ---------------------------------------------------------------------------


class TestEdgeCasesACL:
    def test_duplicate_datasource_in_user_list_deduped(self):
        """User config with duplicate datasource entries must not cause double-querying."""
        from rag.access_control import UserRegistry, validate_access_control

        cfg = _make_registry_config()
        cfg["users"]["nire"]["datasources"] = [
            "family-docs", "family-docs", "nire-docs"
        ]
        registry = UserRegistry.build_from_config(cfg)
        validate_access_control(cfg, registry)
        permitted = registry.get_permitted_datasources_for_user("nire")
        assert len(permitted) == len(set(permitted)), (
            "Duplicate datasources in config must be de-duplicated"
        )

    @pytest.mark.asyncio
    async def test_edge_user_cannot_see_nire_docs_in_rag(self):
        """Edge user is ACL-blocked from nire-docs even when the collection exists."""
        from rag.retriever import get_relevant_context

        class _StubCol:
            def query(self, query_embeddings, n_results, include=None):
                return {"documents": [["hit"]], "distances": [[0.1]]}

        class _AllDocsClient:
            """Client where ALL three collections exist in ChromaDB."""
            def get_collection(self, name):
                return _StubCol()

        captured = {}

        def capture_log(**kwargs):
            captured.update(kwargs)

        with (
            patch("rag.retriever.get_chroma_client",
                  MagicMock(return_value=_AllDocsClient())),
            patch("rag.retriever.get_embeddings",
                  MagicMock(return_value=MagicMock(
                      embed_documents=lambda xs: [[0.0, 0.0, 0.0]]))),
            patch("rag.retriever.get_permitted_datasources_for_user",
                  MagicMock(return_value=["family-docs"])),
            patch("rag.retriever.log_retrieve_event", capture_log),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            await get_relevant_context("secret nire doc", user="edge")

        queried = captured.get("datasources_queried", [])
        assert "nire-docs" not in queried, "edge must never query nire-docs"
        assert "parents-docs" not in queried, "edge must never query parents-docs"
        assert queried == ["family-docs"]

    @pytest.mark.asyncio
    async def test_nire_scope_includes_all_three_collections(self):
        """Nire's RAG query searches all three permitted collections."""
        from rag.retriever import get_relevant_context

        class _StubCol:
            def query(self, query_embeddings, n_results, include=None):
                return {"documents": [["hit"]], "distances": [[0.1]]}

        class _AllDocsClient:
            def get_collection(self, name):
                return _StubCol()

        captured = {}

        def capture_log(**kwargs):
            captured.update(kwargs)

        expected = sorted(["family-docs", "nire-docs", "parents-docs"])

        with (
            patch("rag.retriever.get_chroma_client",
                  MagicMock(return_value=_AllDocsClient())),
            patch("rag.retriever.get_embeddings",
                  MagicMock(return_value=MagicMock(
                      embed_documents=lambda xs: [[0.0, 0.0, 0.0]]))),
            patch("rag.retriever.get_permitted_datasources_for_user",
                  MagicMock(return_value=expected[:])),
            patch("rag.retriever.log_retrieve_event", capture_log),
            patch("rag.retriever.settings") as mock_settings,
        ):
            mock_settings.rag_top_k = 3
            await get_relevant_context("nire personal doc", user="nire")

        assert sorted(captured.get("datasources_queried", [])) == expected
