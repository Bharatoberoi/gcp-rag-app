"""Pytest defaults: avoid real Vertex/Qdrant during unit regression runs."""

from __future__ import annotations

import os

# Before importing the app
os.environ.setdefault("QDRANT_URL", "memory")
os.environ.setdefault("GCP_PROJECT", "")
os.environ.setdefault("PRODUCTION_MODE", "false")
os.environ.setdefault("DOCS_ENABLED", "true")
os.environ.setdefault("API_KEYS", "")
os.environ.setdefault("RERANKER_ENABLED", "false")

import pytest
from starlette.testclient import TestClient

from app.main import app

from tests.mocks import MockRagPipeline


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_mock_rag(client):
    prev = app.state.rag
    app.state.rag = MockRagPipeline()
    try:
        yield client
    finally:
        app.state.rag = prev
