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


def test_documents_are_available_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = client.get("/documents")

    assert response.status_code == 200
    documents = response.json()
    assert len(documents) == 7
    assert documents[0]["source"] == "langchain_intro.txt"


def test_documents_do_not_initialize_agent(monkeypatch):
    def fail_agent_load():
        raise AssertionError("documents endpoint should not initialize the RAG agent")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(main, "get_agent", fail_agent_load)

    response = client.get("/documents")

    assert response.status_code == 200


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
            raise RuntimeError("raw provider secret detail sk-test-secret-value")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("DEBUG_ERRORS", raising=False)
    monkeypatch.setattr(main, "_agent", FailingAgent())

    response = client.post("/chat", json={"question": "What is RAG?"})

    assert response.status_code == 500
    assert response.json()["detail"] == "RAG request failed. Check server logs for details."


def test_debug_agent_errors_are_redacted(monkeypatch):
    class FailingAgent:
        def query(self, question):
            raise RuntimeError(
                "provider failed with sk-1234567890abcdef at /private/tmp/vector-store"
            )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DEBUG_ERRORS", "true")
    monkeypatch.setattr(main, "_agent", FailingAgent())

    response = client.post("/chat", json={"question": "What is RAG?"})

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "[redacted-secret]" in detail
    assert "[redacted-path]" in detail
    assert "sk-1234567890abcdef" not in detail
    assert "/private/tmp" not in detail


def test_debug_agent_errors_are_truncated(monkeypatch):
    class FailingAgent:
        def query(self, question):
            raise RuntimeError("x" * (main.MAX_DEBUG_ERROR_CHARS + 80))

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("DEBUG_ERRORS", "true")
    monkeypatch.setattr(main, "_agent", FailingAgent())

    response = client.post("/chat", json={"question": "What is RAG?"})

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail.endswith("...[truncated]")
    assert len(detail) == main.MAX_DEBUG_ERROR_CHARS + len("...[truncated]")


def test_debug_sessions_errors_are_redacted(monkeypatch):
    class FailingAgent:
        def list_sessions(self):
            raise RuntimeError("session store leaked hf_abcdefghijklmnop at /Users/demo/db")

    monkeypatch.setenv("ENABLE_DEBUG_ENDPOINTS", "true")
    monkeypatch.setenv("DEBUG_ERRORS", "true")
    monkeypatch.setattr(main, "_agent", FailingAgent())

    response = client.get("/chat/sessions")

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "[redacted-secret]" in detail
    assert "[redacted-path]" in detail
    assert "hf_abcdefghijklmnop" not in detail
    assert "/Users/demo" not in detail


def test_clear_session_requires_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    response = client.delete("/chat/conversation/default")

    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_allowed_origins_rejects_wildcards(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "*,https://app.example.com")

    assert main.get_allowed_origins() == ["https://app.example.com"]
