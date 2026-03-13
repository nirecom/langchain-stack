"""
E2E tests for Phase 4A ingestion pipeline.

Tests run against real ChromaDB and real embeddings model.
Requires: docker compose up chromadb, ruri-v3-310m cached.
"""
import os
import pytest
from pathlib import Path

pytestmark = pytest.mark.asyncio


# --- Single file ingestion tests ---


async def test_ingest_txt(client, fixtures_dir, test_datasource):
    filepath = os.path.join(fixtures_dir, "sample.txt")
    with open(filepath, "rb") as f:
        resp = await client.post(
            "/ingest",
            files={"file": ("sample.txt", f, "text/plain")},
            data={"datasource": test_datasource},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["chunks"] > 0
    assert data["datasource"] == test_datasource


async def test_ingest_md(client, fixtures_dir, test_datasource):
    filepath = os.path.join(fixtures_dir, "sample.md")
    with open(filepath, "rb") as f:
        resp = await client.post(
            "/ingest",
            files={"file": ("sample.md", f, "text/markdown")},
            data={"datasource": test_datasource},
        )
    assert resp.status_code == 200
    assert resp.json()["chunks"] > 0


async def test_ingest_pdf(client, fixtures_dir, test_datasource):
    filepath = os.path.join(fixtures_dir, "sample.pdf")
    if not os.path.exists(filepath):
        pytest.skip("sample.pdf not generated — run create_fixtures.py first")
    with open(filepath, "rb") as f:
        resp = await client.post(
            "/ingest",
            files={"file": ("sample.pdf", f, "application/pdf")},
            data={"datasource": test_datasource},
        )
    assert resp.status_code == 200
    assert resp.json()["chunks"] > 0


async def test_ingest_xlsx(client, fixtures_dir, test_datasource):
    filepath = os.path.join(fixtures_dir, "sample.xlsx")
    if not os.path.exists(filepath):
        pytest.skip("sample.xlsx not generated — run create_fixtures.py first")
    with open(filepath, "rb") as f:
        resp = await client.post(
            "/ingest",
            files={"file": ("sample.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
            data={"datasource": test_datasource},
        )
    assert resp.status_code == 200
    assert resp.json()["chunks"] > 0


async def test_ingest_pptx(client, fixtures_dir, test_datasource):
    filepath = os.path.join(fixtures_dir, "sample.pptx")
    if not os.path.exists(filepath):
        pytest.skip("sample.pptx not generated — run create_fixtures.py first")
    with open(filepath, "rb") as f:
        resp = await client.post(
            "/ingest",
            files={"file": ("sample.pptx", f, "application/vnd.openxmlformats-officedocument.presentationml.presentation")},
            data={"datasource": test_datasource},
        )
    assert resp.status_code == 200
    assert resp.json()["chunks"] > 0


async def test_ingest_docx(client, fixtures_dir, test_datasource):
    filepath = os.path.join(fixtures_dir, "sample.docx")
    if not os.path.exists(filepath):
        pytest.skip("sample.docx not generated — run create_fixtures.py first")
    with open(filepath, "rb") as f:
        resp = await client.post(
            "/ingest",
            files={"file": ("sample.docx", f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            data={"datasource": test_datasource},
        )
    assert resp.status_code == 200
    assert resp.json()["chunks"] > 0


# --- Error handling tests ---


async def test_ingest_unsupported_format(client, test_datasource):
    resp = await client.post(
        "/ingest",
        files={"file": ("test.csv", b"a,b,c\n1,2,3", "text/csv")},
        data={"datasource": test_datasource},
    )
    assert resp.status_code == 400
    assert "Unsupported" in resp.json()["detail"]


# --- Duplicate ingestion test ---


async def test_duplicate_ingestion_no_double_chunks(client, fixtures_dir, test_datasource):
    filepath = os.path.join(fixtures_dir, "sample.txt")

    # First ingestion
    with open(filepath, "rb") as f:
        resp1 = await client.post(
            "/ingest",
            files={"file": ("sample.txt", f, "text/plain")},
            data={"datasource": test_datasource},
        )
    chunks1 = resp1.json()["chunks"]

    # Second ingestion of same file
    with open(filepath, "rb") as f:
        resp2 = await client.post(
            "/ingest",
            files={"file": ("sample.txt", f, "text/plain")},
            data={"datasource": test_datasource},
        )
    chunks2 = resp2.json()["chunks"]

    assert chunks1 == chunks2, "Duplicate ingestion should not double chunk count"

    # Verify actual count in ChromaDB
    from models.chroma import get_or_create_collection
    collection = get_or_create_collection(test_datasource)
    result = collection.get(where={"source": "sample.txt"})
    assert len(result["ids"]) == chunks1


# --- Collection isolation test ---


async def test_collection_isolation(client, fixtures_dir):
    ds_a = "test_isolation_a"
    ds_b = "test_isolation_b"

    try:
        filepath = os.path.join(fixtures_dir, "sample.txt")

        with open(filepath, "rb") as f:
            await client.post(
                "/ingest",
                files={"file": ("sample.txt", f, "text/plain")},
                data={"datasource": ds_a},
            )

        with open(filepath, "rb") as f:
            await client.post(
                "/ingest",
                files={"file": ("other.txt", f, "text/plain")},
                data={"datasource": ds_b},
            )

        from models.chroma import get_or_create_collection

        col_a = get_or_create_collection(ds_a)
        col_b = get_or_create_collection(ds_b)

        docs_a = col_a.get()
        docs_b = col_b.get()

        # Each collection should only have its own documents
        for meta in docs_a["metadatas"]:
            assert meta["datasource"] == ds_a
        for meta in docs_b["metadatas"]:
            assert meta["datasource"] == ds_b
    finally:
        from models.chroma import get_chroma_client
        c = get_chroma_client()
        try:
            c.delete_collection(ds_a)
        except Exception:
            pass
        try:
            c.delete_collection(ds_b)
        except Exception:
            pass


# --- Collection deletion test ---


async def test_delete_collection(client, fixtures_dir, test_datasource):
    filepath = os.path.join(fixtures_dir, "sample.txt")

    with open(filepath, "rb") as f:
        await client.post(
            "/ingest",
            files={"file": ("sample.txt", f, "text/plain")},
            data={"datasource": test_datasource},
        )

    resp = await client.delete(f"/ingest/{test_datasource}")
    assert resp.status_code == 200
    assert resp.json()["action"] == "deleted"
