"""Regression tests: HTTP contract and auth wiring (RAG mocked)."""

from __future__ import annotations

import io

import pytest
from starlette.testclient import TestClient

from app.config import settings
from app.main import app

from tests.mocks import MockRagPipeline


def test_health_ok(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "qdrant" in body and "llm" in body


def test_root_redirects_to_ui(client: TestClient):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 302
    assert r.headers.get("location", "").endswith("/ui/")


def test_ui_static_served(client: TestClient):
    r = client.get("/ui/")
    assert r.status_code == 200
    assert b"Production RAG" in r.content


def test_openapi_when_docs_enabled(client: TestClient):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    data = r.json()
    assert "/v1/query" in str(data.get("paths", {}))


@pytest.mark.usefixtures("client_mock_rag")
def test_ingest_multipart(client: TestClient):
    r = client.post(
        "/v1/ingest",
        files={"file": ("note.txt", io.BytesIO(b"hello world"), "text/plain")},
    )
    assert r.status_code == 200
    j = r.json()
    assert j["document"] == "note.txt"
    assert j["chunks_indexed"] == 2


@pytest.mark.usefixtures("client_mock_rag")
def test_query_json(client: TestClient):
    r = client.post("/v1/query", json={"question": "What is RAG?", "top_k": 3})
    assert r.status_code == 200
    j = r.json()
    assert "Echo:" in j["answer"]
    assert len(j["sources"]) == 1
    assert j["sources"][0]["source_document"] == "mock.pdf"


@pytest.mark.usefixtures("client_mock_rag")
def test_delete_document(client: TestClient):
    r = client.delete("/v1/documents/some%2Fdoc.pdf")
    assert r.status_code == 200


def test_query_empty_question(client: TestClient):
    r = client.post("/v1/query", json={"question": "   ", "top_k": 5})
    assert r.status_code == 400


def test_query_rejects_out_of_range_top_k(client: TestClient):
    r = client.post("/v1/query", json={"question": "What is RAG?", "top_k": 1000})
    assert r.status_code == 422


def test_ingest_empty_file(client: TestClient):
    r = client.post(
        "/v1/ingest",
        files={"file": ("empty.txt", io.BytesIO(b""), "text/plain")},
    )
    assert r.status_code == 400


def test_ingest_rejects_oversized_file(client: TestClient):
    prev = settings.max_upload_bytes
    try:
        settings.max_upload_bytes = 4
        r = client.post(
            "/v1/ingest",
            files={"file": ("too-big.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert r.status_code == 413
    finally:
        settings.max_upload_bytes = prev


def test_api_key_rejects_missing_header(client: TestClient):
    prev = settings.api_keys
    try:
        settings.api_keys = "integration-test-key-xyz"
        r = client.post("/v1/query", json={"question": "hi", "top_k": 1})
        assert r.status_code == 401
    finally:
        settings.api_keys = prev


def test_api_key_allows_request_with_mock_rag(client: TestClient):
    prev = settings.api_keys
    prev_rag = app.state.rag
    try:
        settings.api_keys = "good-key"
        app.state.rag = MockRagPipeline()
        r = client.post(
            "/v1/query",
            json={"question": "hi", "top_k": 1},
            headers={"X-API-Key": "good-key"},
        )
        assert r.status_code == 200
    finally:
        settings.api_keys = prev
        app.state.rag = prev_rag
