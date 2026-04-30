"""
E2E tests for Phase 4B: Access control & data protection.

Requires langchain-api container running at localhost:8100.
Tests require INGEST_API_KEY and CHAT_API_KEY to be set in the container.

Test groups:
  - TestIngestAuth: Bearer token authentication on ingest endpoints
  - TestChatAuth: Bearer token authentication on /v1/chat/completions
  - TestDatasourceValidation: access_control.yaml datasource whitelist
  - TestDryRun: dry-run mode returns chunk metadata without DB write
  - TestAuditLog: ingest operations produce audit log entries
  - TestDataIsolation: cross-datasource contamination check
"""
import os
import time
import pytest
import httpx

pytestmark = pytest.mark.e2e

# Per-user chat tokens — mirror the new user-based ACL design.
os.environ.setdefault("CHAT_API_KEY_KYOKO", "test-kyoko-key")
os.environ.setdefault("CHAT_API_KEY_NIRE", "test-nire-key")
os.environ.setdefault("CHAT_API_KEY_EDGE", "test-edge-key")
os.environ.setdefault("CHAT_API_KEY_LUTE", "test-lute-key")
os.environ.setdefault("INGEST_API_KEY", "test-ingest-key")

BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8100")
INGEST_API_KEY = os.getenv("TEST_INGEST_API_KEY",
                           os.environ["INGEST_API_KEY"])
DS_FAMILY = "family-docs"
DS_NIRE = "nire-docs"
DS_PARENTS = "parents-docs"
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures-phase4a")


def _ingest_headers():
    return {"Authorization": f"Bearer {INGEST_API_KEY}"}


def _chat_headers(user: str = "nire"):
    """Return Bearer headers for the given user's chat token."""
    token = os.environ.get(f"CHAT_API_KEY_{user.upper()}", "")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def wait_for_server():
    """Wait until the server is reachable (max 120s)."""
    for _ in range(120):
        try:
            r = httpx.get(f"{BASE_URL}/health", timeout=1)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    pytest.fail("Server not reachable after 120s")


@pytest.fixture
def client():
    with httpx.Client(base_url=BASE_URL, timeout=30) as c:
        yield c


@pytest.fixture(autouse=True)
def cleanup(client):
    """Delete test datasources after each test."""
    yield
    for ds in [DS_FAMILY, DS_NIRE, DS_PARENTS, "unregistered-ds"]:
        client.delete(f"/ingest/{ds}", headers=_ingest_headers())


def _ingest_txt(client, datasource, filename="sample.txt"):
    """Helper: ingest a text fixture into the given datasource."""
    with open(os.path.join(FIXTURES_DIR, filename), "rb") as fp:
        r = client.post(
            "/ingest",
            files={"file": (filename, fp, "text/plain")},
            data={"datasource": datasource},
            headers=_ingest_headers(),
        )
    return r


# ── TestIngestAuth ──────────────────────────────────────────────


class TestIngestAuth:
    def test_valid_token(self, wait_for_server, client):
        """Valid INGEST_API_KEY → 200."""
        r = _ingest_txt(client, DS_FAMILY)
        assert r.status_code == 200

    def test_missing_token(self, client):
        """No Authorization header → 401."""
        with open(os.path.join(FIXTURES_DIR, "sample.txt"), "rb") as fp:
            r = client.post(
                "/ingest",
                files={"file": ("sample.txt", fp, "text/plain")},
                data={"datasource": DS_FAMILY},
            )
        assert r.status_code == 401

    def test_invalid_token(self, client):
        """Wrong token → 401."""
        with open(os.path.join(FIXTURES_DIR, "sample.txt"), "rb") as fp:
            r = client.post(
                "/ingest",
                files={"file": ("sample.txt", fp, "text/plain")},
                data={"datasource": DS_FAMILY},
                headers={"Authorization": "Bearer wrong-key"},
            )
        assert r.status_code == 401

    def test_delete_requires_auth(self, client):
        """DELETE /ingest/{ds} without token → 401."""
        r = client.delete(f"/ingest/{DS_FAMILY}")
        assert r.status_code == 401

    def test_list_requires_auth(self, client):
        """GET /ingest/{ds} without token → 401."""
        r = client.get(f"/ingest/{DS_FAMILY}")
        assert r.status_code == 401

    def test_batch_requires_auth(self, client):
        """POST /ingest/batch without token → 401."""
        r = client.post("/ingest/batch", data={"datasource": DS_FAMILY})
        assert r.status_code == 401

    def test_delete_file_requires_auth(self, client):
        """DELETE /ingest/{ds}/{file} without token → 401."""
        r = client.delete(f"/ingest/{DS_FAMILY}/sample.txt")
        assert r.status_code == 401


# ── TestChatAuth ────────────────────────────────────────────────


class TestChatAuth:
    @pytest.mark.parametrize("user", ["kyoko", "nire", "edge", "lute"])
    def test_valid_token(self, wait_for_server, client, user):
        """Each valid per-user chat token → 200 (or model error, but not 401)."""
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "judge-chain",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers=_chat_headers(user),
        )
        # May fail for other reasons (LLM down), but must not be 401
        assert r.status_code != 401

    def test_missing_token(self, client):
        """No Authorization header → 401."""
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "judge-chain",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert r.status_code == 401

    def test_invalid_token(self, client):
        """Wrong token → 401."""
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "judge-chain",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert r.status_code == 401

    def test_stream_requires_auth(self, client):
        """Streaming chat request without token → 401."""
        r = client.post(
            "/v1/chat/completions",
            json={
                "model": "judge-chain",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )
        assert r.status_code == 401


# ── TestDatasourceValidation ────────────────────────────────────


class TestDatasourceValidation:
    def test_registered_datasource_ok(self, wait_for_server, client):
        """Ingesting to a registered datasource succeeds."""
        r = _ingest_txt(client, DS_FAMILY)
        assert r.status_code == 200

    def test_unregistered_datasource_rejected(self, client):
        """Ingesting to an unregistered datasource → 403."""
        with open(os.path.join(FIXTURES_DIR, "sample.txt"), "rb") as fp:
            r = client.post(
                "/ingest",
                files={"file": ("sample.txt", fp, "text/plain")},
                data={"datasource": "unregistered-ds"},
                headers=_ingest_headers(),
            )
        assert r.status_code == 403

    def test_batch_unregistered_rejected(self, client):
        """Batch ingest to an unregistered datasource → 403."""
        r = client.post(
            "/ingest/batch",
            data={"datasource": "unregistered-ds"},
            headers=_ingest_headers(),
        )
        assert r.status_code == 403

    def test_delete_unregistered_rejected(self, client):
        """Delete of an unregistered datasource → 403."""
        r = client.delete(
            "/ingest/unregistered-ds",
            headers=_ingest_headers(),
        )
        assert r.status_code == 403


# ── TestDryRun ──────────────────────────────────────────────────


class TestDryRun:
    def test_dry_run_returns_chunks(self, wait_for_server, client):
        """dry_run=true returns chunk metadata."""
        with open(os.path.join(FIXTURES_DIR, "sample.txt"), "rb") as fp:
            r = client.post(
                "/ingest",
                files={"file": ("sample.txt", fp, "text/plain")},
                data={"datasource": DS_FAMILY, "dry_run": "true"},
                headers=_ingest_headers(),
            )
        assert r.status_code == 200
        body = r.json()
        assert body["dry_run"] is True
        assert body["total_chunks"] >= 1
        assert len(body["chunks"]) >= 1
        chunk = body["chunks"][0]
        assert "chunk_index" in chunk
        assert "char_count" in chunk
        assert "preview" in chunk

    def test_dry_run_no_side_effect(self, client):
        """dry_run=true does not write to ChromaDB."""
        with open(os.path.join(FIXTURES_DIR, "sample.txt"), "rb") as fp:
            client.post(
                "/ingest",
                files={"file": ("sample.txt", fp, "text/plain")},
                data={"datasource": DS_FAMILY, "dry_run": "true"},
                headers=_ingest_headers(),
            )
        # Listing the datasource should show no files (collection may not exist)
        r = client.get(
            f"/ingest/{DS_FAMILY}",
            headers=_ingest_headers(),
        )
        if r.status_code == 200:
            assert r.json()["files"] == []
        else:
            # 404 means collection doesn't exist — also correct
            assert r.status_code == 404

    def test_dry_run_idempotent(self, client):
        """Same file dry-run twice produces identical results."""
        results = []
        for _ in range(2):
            with open(os.path.join(FIXTURES_DIR, "sample.txt"), "rb") as fp:
                r = client.post(
                    "/ingest",
                    files={"file": ("sample.txt", fp, "text/plain")},
                    data={"datasource": DS_FAMILY, "dry_run": "true"},
                    headers=_ingest_headers(),
                )
            assert r.status_code == 200
            results.append(r.json())
        assert results[0]["total_chunks"] == results[1]["total_chunks"]
        assert len(results[0]["chunks"]) == len(results[1]["chunks"])


# ── TestAuditLog ────────────────────────────────────────────────


class TestAuditLog:
    def test_ingest_creates_audit_entry(self, wait_for_server, client):
        """Successful ingest creates an audit log entry."""
        _ingest_txt(client, DS_FAMILY)
        r = client.get(
            "/audit/recent",
            headers=_ingest_headers(),
        )
        assert r.status_code == 200
        events = r.json()["events"]
        # Find the most recent ingest event for datasource_a
        matches = [
            e for e in events
            if e["datasource"] == DS_FAMILY
            and e["action"] == "ingest"
            and e["status"] == "ok"
        ]
        assert len(matches) >= 1
        entry = matches[-1]
        assert entry["filename"] == "sample.txt"
        assert entry["chunks"] >= 1
        assert "timestamp" in entry

    def test_delete_creates_audit_entry(self, client):
        """Deleting a datasource creates an audit log entry."""
        _ingest_txt(client, DS_FAMILY)
        client.delete(f"/ingest/{DS_FAMILY}", headers=_ingest_headers())
        r = client.get("/audit/recent", headers=_ingest_headers())
        assert r.status_code == 200
        events = r.json()["events"]
        matches = [
            e for e in events
            if e["datasource"] == DS_FAMILY
            and e["action"] == "delete"
        ]
        assert len(matches) >= 1

    def test_error_logged(self, client):
        """Failed ingest (unregistered datasource) creates error audit entry."""
        with open(os.path.join(FIXTURES_DIR, "sample.txt"), "rb") as fp:
            client.post(
                "/ingest",
                files={"file": ("sample.txt", fp, "text/plain")},
                data={"datasource": "unregistered-ds"},
                headers=_ingest_headers(),
            )
        r = client.get("/audit/recent", headers=_ingest_headers())
        assert r.status_code == 200
        events = r.json()["events"]
        matches = [
            e for e in events
            if e["datasource"] == "unregistered-ds"
            and e["status"] == "error"
        ]
        assert len(matches) >= 1

    def test_audit_requires_auth(self, client):
        """GET /audit/recent without token → 401."""
        r = client.get("/audit/recent")
        assert r.status_code == 401

    def test_audit_recent_ordering(self, client):
        """Audit events should be ordered by timestamp (most recent last)."""
        _ingest_txt(client, DS_FAMILY, filename="sample.txt")
        _ingest_txt(client, DS_FAMILY, filename="sample.md")
        r = client.get("/audit/recent", headers=_ingest_headers())
        assert r.status_code == 200
        events = r.json()["events"]
        timestamps = [e["timestamp"] for e in events]
        assert timestamps == sorted(timestamps)


# ── TestDataIsolation ───────────────────────────────────────────


class TestDataIsolation:
    def test_ingest_to_a_not_in_b(self, wait_for_server, client):
        """File ingested to datasource_a must not appear in datasource_b."""
        _ingest_txt(client, DS_FAMILY)
        r = client.get(
            f"/ingest/{DS_NIRE}",
            headers=_ingest_headers(),
        )
        if r.status_code == 200:
            filenames = [f["filename"] for f in r.json()["files"]]
            assert "sample.txt" not in filenames
        else:
            # 404 means collection doesn't exist — isolation confirmed
            assert r.status_code == 404

    def test_ingest_to_b_not_in_a(self, client):
        """File ingested to datasource_b must not appear in datasource_a."""
        _ingest_txt(client, DS_NIRE, filename="sample.md")
        r = client.get(
            f"/ingest/{DS_FAMILY}",
            headers=_ingest_headers(),
        )
        if r.status_code == 200:
            filenames = [f["filename"] for f in r.json()["files"]]
            assert "sample.md" not in filenames
        else:
            assert r.status_code == 404


# ── TestIdempotency ─────────────────────────────────────────────


class TestIdempotency:
    def test_ingest_duplicate_replaces(self, wait_for_server, client):
        """Ingesting same file twice does not double the chunk count."""
        _ingest_txt(client, DS_FAMILY)
        r1 = client.get(f"/ingest/{DS_FAMILY}", headers=_ingest_headers())
        count1 = r1.json()["files"][0]["chunk_count"]

        _ingest_txt(client, DS_FAMILY)
        r2 = client.get(f"/ingest/{DS_FAMILY}", headers=_ingest_headers())
        count2 = r2.json()["files"][0]["chunk_count"]

        assert count1 == count2
        assert len(r2.json()["files"]) == 1

    def test_delete_datasource_twice(self, client):
        """Deleting same datasource twice: first succeeds, second errors gracefully."""
        _ingest_txt(client, DS_FAMILY)
        r1 = client.delete(f"/ingest/{DS_FAMILY}", headers=_ingest_headers())
        assert r1.status_code == 200

        r2 = client.delete(f"/ingest/{DS_FAMILY}", headers=_ingest_headers())
        # Second delete may return 200 or 500 depending on ChromaDB behavior
        # but should not crash the server
        assert r2.status_code in (200, 404, 500)
