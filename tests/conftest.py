"""
Test fixtures for Phase 4A ingestion tests.

Requires:
- ChromaDB running at localhost:8200 (or CHROMA_HOST/CHROMA_PORT env vars)
- ruri-v3-310m model cached at data/hf_cache/
"""
import os
import sys
import pytest
from httpx import AsyncClient, ASGITransport

# Add app directory to path so imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

# Override settings before importing app
os.environ.setdefault("CHROMA_HOST", "localhost")
os.environ.setdefault("CHROMA_PORT", "8200")
os.environ.setdefault("LITELLM_PROXY_URL", "http://localhost:4000/v1")


@pytest.fixture
def fixtures_dir():
    return os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
async def client():
    from main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def test_datasource():
    """Provide a test datasource name and clean up after test."""
    name = "test_phase4a"
    yield name
    # Cleanup: delete collection after test
    try:
        from models.chroma import get_chroma_client
        client = get_chroma_client()
        client.delete_collection(name=name)
    except Exception:
        pass
