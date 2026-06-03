#!/usr/bin/env python3
"""Local API smoke tests for the LangChain RAG FastAPI service."""

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_health_check_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["openai_configured"] is False


def test_chat_requires_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = client.post("/chat", json={"question": "What is RAG?"})

    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_debug_endpoints_are_hidden_by_default(monkeypatch):
    monkeypatch.delenv("ENABLE_DEBUG_ENDPOINTS", raising=False)

    assert client.get("/debug/langsmith").status_code == 404
    assert client.get("/chat/sessions").status_code == 404


def test_request_validation_rejects_empty_question():
    response = client.post("/chat", json={"question": ""})

    assert response.status_code == 422
