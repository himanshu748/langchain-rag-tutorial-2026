#!/usr/bin/env python3
"""Local API smoke tests for the LangChain RAG FastAPI service."""

from fastapi.testclient import TestClient

import main
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


def test_blank_question_is_rejected_before_agent_call(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    response = client.post("/chat", json={"question": "   "})

    assert response.status_code == 422
    assert "blank" in response.json()["detail"]


def test_conversation_rejects_unsafe_session_id(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    response = client.post(
        "/chat/conversation",
        json={"question": "What is RAG?", "session_id": "../shared"},
    )

    assert response.status_code == 422


def test_agent_errors_are_sanitized_by_default(monkeypatch):
    class FailingAgent:
        def query(self, question):
            raise RuntimeError("raw provider secret detail")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("DEBUG_ERRORS", raising=False)
    monkeypatch.setattr(main, "_agent", FailingAgent())

    response = client.post("/chat", json={"question": "What is RAG?"})

    assert response.status_code == 500
    assert response.json()["detail"] == "RAG request failed. Check server logs for details."


def test_clear_session_requires_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = client.delete("/chat/conversation/default")

    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_allowed_origins_rejects_wildcards(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "*,https://app.example.com")

    assert main.get_allowed_origins() == ["https://app.example.com"]
