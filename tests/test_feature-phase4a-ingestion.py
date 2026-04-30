"""
E2E tests for Phase 4A ingestion endpoints.

Requires langchain-api container running at localhost:8100.
Binary fixtures (pdf, xlsx, pptx, docx) require create_fixtures.py to be run first.
"""
import os
import time
import pytest
import httpx

pytestmark = pytest.mark.e2e

BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8100")
INGEST_API_KEY = os.getenv("TEST_INGEST_API_KEY", "test-ingest-key")
DATASOURCE = "test-pytest"
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
    pytest.fail("Server not reachable after 30s")


@pytest.fixture
def client():
    with httpx.Client(base_url=BASE_URL, timeout=30) as c:
        yield c


@pytest.fixture(autouse=True)
def cleanup(client):
    """Delete test datasource after each test."""
    yield
    client.delete(f"/ingest/{DATASOURCE}", headers=_ingest_headers())


class TestHealth:
    def test_health(self, wait_for_server, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


class TestIngestUpload:
    def test_ingest_txt(self, client):
        with open(os.path.join(FIXTURES_DIR, "sample.txt"), "rb") as fp:
            r = client.post(
                "/ingest",
                files={"file": ("sample.txt", fp, "text/plain")},
                data={"datasource": DATASOURCE},
                headers=_ingest_headers(),
            )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["filename"] == "sample.txt"
        assert body["datasource"] == DATASOURCE
        assert body["chunks"] >= 1

    def test_ingest_md(self, client):
        with open(os.path.join(FIXTURES_DIR, "sample.md"), "rb") as fp:
            r = client.post(
                "/ingest",
                files={"file": ("sample.md", fp, "text/markdown")},
                data={"datasource": DATASOURCE},
                headers=_ingest_headers(),
            )
        assert r.status_code == 200
        assert r.json()["chunks"] >= 1

    def test_ingest_pdf(self, client):
        path = os.path.join(FIXTURES_DIR, "sample.pdf")
        if not os.path.exists(path):
            pytest.skip("sample.pdf not generated — run create_fixtures.py first")
        with open(path, "rb") as fp:
            r = client.post(
                "/ingest",
                files={"file": ("sample.pdf", fp, "application/pdf")},
                data={"datasource": DATASOURCE},
                headers=_ingest_headers(),
            )
        assert r.status_code == 200
        assert r.json()["chunks"] >= 1

    def test_ingest_xlsx(self, client):
        path = os.path.join(FIXTURES_DIR, "sample.xlsx")
        if not os.path.exists(path):
            pytest.skip("sample.xlsx not generated — run create_fixtures.py first")
        with open(path, "rb") as fp:
            r = client.post(
                "/ingest",
                files={"file": ("sample.xlsx", fp, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
                data={"datasource": DATASOURCE},
                headers=_ingest_headers(),
            )
        assert r.status_code == 200
        assert r.json()["chunks"] >= 1

    def test_ingest_pptx(self, client):
        path = os.path.join(FIXTURES_DIR, "sample.pptx")
        if not os.path.exists(path):
            pytest.skip("sample.pptx not generated — run create_fixtures.py first")
        with open(path, "rb") as fp:
            r = client.post(
                "/ingest",
                files={"file": ("sample.pptx", fp, "application/vnd.openxmlformats-officedocument.presentationml.presentation")},
                data={"datasource": DATASOURCE},
                headers=_ingest_headers(),
            )
        assert r.status_code == 200
        assert r.json()["chunks"] >= 1

    def test_ingest_docx(self, client):
        path = os.path.join(FIXTURES_DIR, "sample.docx")
        if not os.path.exists(path):
            pytest.skip("sample.docx not generated — run create_fixtures.py first")
        with open(path, "rb") as fp:
            r = client.post(
                "/ingest",
                files={"file": ("sample.docx", fp, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
                data={"datasource": DATASOURCE},
                headers=_ingest_headers(),
            )
        assert r.status_code == 200
        assert r.json()["chunks"] >= 1

    def test_ingest_duplicate(self, client):
        """Re-ingesting the same file replaces existing chunks."""
        with open(os.path.join(FIXTURES_DIR, "sample.txt"), "rb") as fp:
            r1 = client.post(
                "/ingest",
                files={"file": ("sample.txt", fp, "text/plain")},
                data={"datasource": DATASOURCE},
                headers=_ingest_headers(),
            )
        with open(os.path.join(FIXTURES_DIR, "sample.txt"), "rb") as fp:
            r2 = client.post(
                "/ingest",
                files={"file": ("sample.txt", fp, "text/plain")},
                data={"datasource": DATASOURCE},
                headers=_ingest_headers(),
            )
        assert r1.json()["chunks"] == r2.json()["chunks"]


class TestIngestError:
    def test_unsupported_extension(self, client, tmp_path):
        f = tmp_path / "bad.exe"
        f.write_bytes(b"\x00\x01\x02")
        with open(f, "rb") as fp:
            r = client.post(
                "/ingest",
                files={"file": ("bad.exe", fp, "application/octet-stream")},
                data={"datasource": DATASOURCE},
                headers=_ingest_headers(),
            )
        assert r.status_code == 400
        assert "Unsupported" in r.json()["detail"]

    def test_missing_datasource(self, client, tmp_path):
        f = tmp_path / "no_ds.txt"
        f.write_text("No datasource.")
        with open(f, "rb") as fp:
            r = client.post(
                "/ingest",
                files={"file": ("no_ds.txt", fp, "text/plain")},
                headers=_ingest_headers(),
            )
        assert r.status_code == 422


class TestIngestDelete:
    def test_delete_collection(self, client):
        with open(os.path.join(FIXTURES_DIR, "sample.txt"), "rb") as fp:
            client.post(
                "/ingest",
                files={"file": ("sample.txt", fp, "text/plain")},
                data={"datasource": DATASOURCE},
                headers=_ingest_headers(),
            )
        r = client.delete(f"/ingest/{DATASOURCE}", headers=_ingest_headers())
        assert r.status_code == 200
        assert r.json()["action"] == "deleted"
