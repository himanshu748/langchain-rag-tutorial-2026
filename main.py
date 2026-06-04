"""
LangChain RAG Tutorial API
FastAPI application for demonstrating RAG with LangChain v1.x.

Deploy on Render: https://render.com
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os
import re
from dotenv import load_dotenv
from knowledge_base import list_sample_documents

# Load environment variables
load_dotenv()

DEFAULT_ALLOWED_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
MAX_QUESTION_CHARS = 4_000
MAX_SESSION_ID_CHARS = 96
MAX_DEBUG_ERROR_CHARS = 240
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$")
SECRET_PATTERN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}|hf_[A-Za-z0-9]{8,}|gh[pousr]_[A-Za-z0-9_]{8,}|AIza[A-Za-z0-9_-]{8,})\b"
)
LOCAL_PATH_PATTERN = re.compile(r"(?:/private|/Users|/var|/tmp)/[^\s'\"<>]+")


def env_flag(name: str, default: bool = False) -> bool:
    """Parse common truthy environment variable values."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_has_value(name: str) -> bool:
    """Return True only when an environment variable has non-blank content."""
    return bool(os.getenv(name, "").strip())


def get_allowed_origins() -> list[str]:
    """Return explicit CORS origins from ALLOWED_ORIGINS."""
    raw = os.getenv("ALLOWED_ORIGINS", ",".join(DEFAULT_ALLOWED_ORIGINS))
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    safe_origins = [
        origin
        for origin in origins
        if origin != "*" and origin.startswith(("http://", "https://"))
    ]
    return safe_origins or DEFAULT_ALLOWED_ORIGINS


def validate_session_id(session_id: str) -> str:
    """Validate and normalize session IDs before they reach checkpoint storage."""
    normalized = session_id.strip()
    if not SESSION_ID_PATTERN.fullmatch(normalized):
        raise HTTPException(
            status_code=422,
            detail=(
                "session_id must start with a letter or number and contain only "
                "letters, numbers, dots, underscores, colons, or hyphens."
            ),
        )
    return normalized


def sanitize_question(question: str) -> str:
    normalized = question.strip()
    if not normalized:
        raise HTTPException(status_code=422, detail="question must not be blank")
    return normalized

# Initialize FastAPI app
app = FastAPI(
    title="LangChain RAG Tutorial API",
    description="""
## 🚀 LangChain RAG Chatbot API

A production-ready RAG (Retrieval Augmented Generation) API built with:
- **LangChain v1.2.4** (January 2026)
- **FastAPI** for high-performance API
- **Chroma** for vector storage
- **GPT-4o-mini** for responses

### Features
- 💬 Single-turn chat queries
- 🔄 Multi-turn conversations with memory
- 📚 Knowledge base about LangChain, RAG, and AI concepts

### Tutorial Source
Based on [FutureSmart.ai RAG Tutorial](https://blog.futuresmart.ai/langchain-rag-from-basics-to-production-ready-rag-chatbot)
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    question: str = Field(
        ...,
        description="The question to ask the RAG agent",
        min_length=1,
        max_length=MAX_QUESTION_CHARS,
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "What is RAG and how does it work?"},
                {"question": "What are the popular vector databases?"},
                {"question": "Explain LangChain in simple terms"},
            ]
        }
    }


class ConversationRequest(BaseModel):
    """Request model for conversation endpoint with session tracking."""
    question: str = Field(
        ...,
        description="The question to ask",
        min_length=1,
        max_length=MAX_QUESTION_CHARS,
    )
    session_id: str = Field(
        default="default",
        description="Session ID for conversation memory",
        min_length=1,
        max_length=MAX_SESSION_ID_CHARS,
        pattern=SESSION_ID_PATTERN.pattern,
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"question": "What is LangChain?", "session_id": "user-123"},
                {"question": "How does it relate to LangGraph?", "session_id": "user-123"},
            ]
        }
    }


class ChatResponse(BaseModel):
    """Response model for chat endpoints."""
    answer: str = Field(..., description="The agent's response")
    session_id: Optional[str] = Field(None, description="Session ID if conversation mode")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    version: str
    openai_configured: bool
    langsmith_configured: bool
    debug_endpoints_enabled: bool


class DocumentInfo(BaseModel):
    """Document information."""
    content: str
    source: str
    page: int


# Lazy load agent to avoid startup delays
_agent = None

def get_agent():
    """Lazy load the RAG agent."""
    global _agent
    if _agent is None:
        from rag_agent import get_agent as create_agent
        _agent = create_agent()
    return _agent


def require_openai_key() -> None:
    if not env_has_value("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key is not configured. Set OPENAI_API_KEY before making RAG queries.",
        )


def internal_error(exc: Exception) -> HTTPException:
    if env_flag("DEBUG_ERRORS"):
        detail = SECRET_PATTERN.sub("[redacted-secret]", str(exc))
        detail = LOCAL_PATH_PATTERN.sub("[redacted-path]", detail)
        if len(detail) > MAX_DEBUG_ERROR_CHARS:
            detail = detail[:MAX_DEBUG_ERROR_CHARS] + "...[truncated]"
        return HTTPException(status_code=500, detail=detail)
    return HTTPException(status_code=500, detail="RAG request failed. Check server logs for details.")


# Endpoints
@app.get("/", tags=["Info"])
async def root():
    """API root - returns service information."""
    return {
        "service": "LangChain RAG Tutorial API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "conversation": "/chat/conversation",
            "documents": "/documents"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        service="LangChain RAG API",
        version="1.0.0",
        openai_configured=env_has_value("OPENAI_API_KEY"),
        langsmith_configured=env_flag("LANGSMITH_TRACING")
        and env_has_value("LANGSMITH_API_KEY"),
        debug_endpoints_enabled=env_flag("ENABLE_DEBUG_ENDPOINTS"),
    )


@app.get("/debug/langsmith", tags=["Debug"])
async def debug_langsmith():
    """Debug endpoint to verify LangSmith configuration."""
    if not env_flag("ENABLE_DEBUG_ENDPOINTS"):
        raise HTTPException(status_code=404, detail="Not found")
    return {
        "LANGSMITH_TRACING": os.getenv("LANGSMITH_TRACING", "NOT SET"),
        "LANGSMITH_ENDPOINT": os.getenv("LANGSMITH_ENDPOINT", "NOT SET"),
        "LANGSMITH_API_KEY": "SET" if os.getenv("LANGSMITH_API_KEY") else "NOT SET",
        "LANGSMITH_PROJECT": os.getenv("LANGSMITH_PROJECT", "NOT SET"),
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Single-turn RAG query.
    
    Send a question and get an answer based on the knowledge base.
    This endpoint does NOT maintain conversation history.
    """
    require_openai_key()
    question = sanitize_question(request.question)
    
    try:
        agent = get_agent()
        answer = agent.query(question)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise internal_error(e)


@app.post("/chat/conversation", response_model=ChatResponse, tags=["Chat"])
async def conversation(request: ConversationRequest):
    """
    Multi-turn conversation with memory.
    
    Send a question with a session_id to maintain conversation context.
    Use the same session_id for follow-up questions.
    
    Example flow:
    1. {"question": "What is LangChain?", "session_id": "user-123"}
    2. {"question": "How does it work?", "session_id": "user-123"}  # Remembers context
    """
    require_openai_key()
    question = sanitize_question(request.question)
    session_id = validate_session_id(request.session_id)
    
    try:
        agent = get_agent()
        answer = agent.chat(question, session_id)
        return ChatResponse(answer=answer, session_id=session_id)
    except Exception as e:
        raise internal_error(e)


@app.get("/documents", response_model=list[DocumentInfo], tags=["Knowledge Base"])
async def list_documents():
    """List all built-in tutorial documents without requiring provider credentials."""
    return list_sample_documents(truncate=False)


@app.delete("/chat/conversation/{session_id}", tags=["Chat"])
async def clear_session(session_id: str):
    """
    Clear conversation history for a specific session.
    
    Use this to reset a user's conversation context.
    After clearing, the next message from this session will start fresh.
    """
    require_openai_key()
    try:
        session_id = validate_session_id(session_id)
        agent = get_agent()
        success = agent.clear_session(session_id)
        if success:
            return {"message": f"Session '{session_id}' cleared successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or could not be cleared")
    except HTTPException:
        raise
    except Exception as e:
        raise internal_error(e)


@app.get("/chat/sessions", tags=["Chat"])
async def list_sessions():
    """List all active session IDs (for debugging/admin purposes)."""
    if not env_flag("ENABLE_DEBUG_ENDPOINTS"):
        raise HTTPException(status_code=404, detail="Not found")
    try:
        agent = get_agent()
        sessions = agent.list_sessions()
        return {"sessions": sessions, "count": len(sessions)}
    except Exception as e:
        raise internal_error(e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
