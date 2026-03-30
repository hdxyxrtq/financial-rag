"""FastAPI REST API 测试 — 使用 TestClient 同步测试。"""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.deps import get_pipeline, get_store

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _mock_store():
    store = MagicMock()
    store.get_stats.return_value = {"collection_name": "financial_docs", "document_count": 42}
    store.get_all_ids.return_value = ["doc1_abc", "doc2_def"]
    store.get_all_documents.return_value = {"doc1_abc": "content1", "doc2_def": "content2"}
    store.delete_by_ids.return_value = None
    return store


@pytest.fixture()
def _mock_pipeline():
    pipeline = MagicMock()
    pipeline.aquery = AsyncMock(
        return_value={
            "answer": "GDP 是国内生产总值。",
            "sources": [
                {"content": "GDP 定义...", "score": 0.95, "metadata": {"source": "经济学基础"}},
            ],
        }
    )
    return pipeline


@pytest.fixture()
def client(_mock_store, _mock_pipeline):
    app.dependency_overrides[get_store] = lambda: _mock_store
    app.dependency_overrides[get_pipeline] = lambda: _mock_pipeline
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GET /api/v1/health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"
        assert data["indexed_chunks"] == 42


# ---------------------------------------------------------------------------
# POST /api/v1/query
# ---------------------------------------------------------------------------


class TestQuery:
    def test_query_normal(self, client):
        resp = client.post("/api/v1/query", json={"question": "什么是GDP？"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert len(data["sources"]) > 0
        assert data["latency_ms"] > 0

    def test_query_empty_question_returns_422(self, client):
        resp = client.post("/api/v1/query", json={"question": ""})
        assert resp.status_code == 422

    def test_query_missing_question_returns_422(self, client):
        resp = client.post("/api/v1/query", json={})
        assert resp.status_code == 422

    def test_query_with_history(self, client):
        resp = client.post(
            "/api/v1/query",
            json={
                "question": "那它呢？",
                "chat_history": [
                    {"role": "user", "content": "什么是GDP？"},
                    {"role": "assistant", "content": "GDP是国内生产总值。"},
                ],
            },
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/v1/query/stream
# ---------------------------------------------------------------------------


class TestQueryStream:
    def test_stream_returns_events(self, client, _mock_pipeline):
        _mock_pipeline.astream_query = MagicMock()
        _mock_pipeline.astream_query.return_value = _async_gen(
            [
                {"type": "sources", "sources": [{"content": "test", "score": 0.9, "metadata": {}}]},
                {"type": "answer", "content": "GDP是"},
                {"type": "answer", "content": "国内生产总值"},
            ]
        )
        resp = client.post("/api/v1/query/stream", json={"question": "什么是GDP？"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/v1/documents/upload
# ---------------------------------------------------------------------------


class TestDocumentUpload:
    def test_upload_text_document(self, client):
        content = base64.b64encode("这是一段测试文本。\n\n第二段落内容。".encode()).decode()
        resp = client.post(
            "/api/v1/documents/upload",
            json={"filename": "test.txt", "content": content, "doc_type": "text"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["chunk_count"] > 0

    def test_upload_empty_content_returns_400(self, client):
        content = base64.b64encode(b"   ").decode()
        resp = client.post(
            "/api/v1/documents/upload",
            json={"filename": "empty.txt", "content": content, "doc_type": "text"},
        )
        assert resp.status_code == 400

    def test_upload_missing_filename_returns_422(self, client):
        resp = client.post("/api/v1/documents/upload", json={"content": "dGVzdA==", "doc_type": "text"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/documents/stats
# ---------------------------------------------------------------------------


class TestDocumentStats:
    def test_stats_returns_counts(self, client):
        resp = client.get("/api/v1/documents/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_chunks"] == 42
        assert data["unique_sources"] > 0


# ---------------------------------------------------------------------------
# DELETE /api/v1/documents/{doc_id}
# ---------------------------------------------------------------------------


class TestDocumentDelete:
    def test_delete_existing_document(self, client):
        resp = client.delete("/api/v1/documents/doc1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["deleted_count"] > 0

    def test_delete_nonexistent_document_returns_404(self, client):
        resp = client.delete("/api/v1/documents/nonexistent_id")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _async_gen(items):
    for item in items:
        yield item
