"""
E2E tests for datasource management APIs (file listing and per-file deletion).

Requires langchain-api container running at localhost:8100.
Uses the same fixtures as Phase 4A tests.
"""
import os
import time
import pytest
import httpx

pytestmark = pytest.mark.e2e

BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8100")
INGEST_API_KEY = os.getenv("TEST_INGEST_API_KEY", "test-ingest-key")
DATASOURCE = "test-ds-mgmt"
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures-phase4a")


def _ingest_headers():
    return {"Authorization": f"Bearer {INGEST_API_KEY}"}


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
    """Delete test datasource after each test."""
    yield
    client.delete(f"/ingest/{DATASOURCE}", headers=_ingest_headers())


def _ingest_file(client, filename):
    """Helper: ingest a fixture file into the test datasource."""
    with open(os.path.join(FIXTURES_DIR, filename), "rb") as fp:
        r = client.post(
            "/ingest",
            files={"file": (filename, fp)},
            data={"datasource": DATASOURCE},
            headers=_ingest_headers(),
        )
    assert r.status_code == 200
    return r.json()


class TestDatasourceDetail:
    def test_detail_nonexistent(self, wait_for_server, client):
        # DATASOURCE is registered in access_control.yaml but has no
        # ChromaDB collection yet (cleanup deletes it after each test)
        r = client.get(f"/ingest/{DATASOURCE}", headers=_ingest_headers())
        assert r.status_code == 404

    def test_detail_single_file(self, client):
        _ingest_file(client, "sample.txt")
        r = client.get(f"/ingest/{DATASOURCE}", headers=_ingest_headers())
        assert r.status_code == 200
        body = r.json()
        assert body["datasource"] == DATASOURCE
        assert len(body["files"]) == 1
        assert body["files"][0]["filename"] == "sample.txt"
        assert body["files"][0]["chunk_count"] >= 1

    def test_detail_multiple_files(self, client):
        _ingest_file(client, "sample.txt")
        _ingest_file(client, "sample.md")
        r = client.get(f"/ingest/{DATASOURCE}", headers=_ingest_headers())
        assert r.status_code == 200
        body = r.json()
        filenames = [f["filename"] for f in body["files"]]
        assert "sample.txt" in filenames
        assert "sample.md" in filenames
        assert len(body["files"]) == 2


class TestDeleteFile:
    def test_delete_file(self, client):
        """Delete one file's chunks, verify the other remains."""
        _ingest_file(client, "sample.txt")
        _ingest_file(client, "sample.md")

        r = client.delete(f"/ingest/{DATASOURCE}/sample.txt", headers=_ingest_headers())
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["filename"] == "sample.txt"
        assert body["deleted_chunks"] >= 1

        # Verify only sample.md remains
        r = client.get(f"/ingest/{DATASOURCE}", headers=_ingest_headers())
        assert r.status_code == 200
        filenames = [f["filename"] for f in r.json()["files"]]
        assert "sample.txt" not in filenames
        assert "sample.md" in filenames

    def test_delete_nonexistent_datasource(self, client):
        # Use registered but empty datasource (no ChromaDB collection)
        r = client.delete(f"/ingest/{DATASOURCE}/file.txt", headers=_ingest_headers())
        assert r.status_code == 404

    def test_delete_nonexistent_file(self, client):
        _ingest_file(client, "sample.txt")
        r = client.delete(f"/ingest/{DATASOURCE}/nonexistent.txt", headers=_ingest_headers())
        assert r.status_code == 404
