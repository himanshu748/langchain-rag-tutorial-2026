"""
LangChain RAG Tutorial API
FastAPI Application for demonstrating RAG with LangChain v1.2.4

Deploy on Render: https://render.com
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  Warning: OPENAI_API_KEY not set. Set it before making queries.")

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    question: str = Field(..., description="The question to ask the RAG agent", min_length=1)
    
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
    question: str = Field(..., description="The question to ask", min_length=1)
    session_id: str = Field(default="default", description="Session ID for conversation memory")
    
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
        openai_configured=bool(os.getenv("OPENAI_API_KEY")),
        langsmith_configured=bool(os.getenv("LANGSMITH_TRACING") and os.getenv("LANGSMITH_API_KEY"))
    )


@app.get("/debug/langsmith", tags=["Debug"])
async def debug_langsmith():
    """Debug endpoint to verify LangSmith configuration."""
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
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Set OPENAI_API_KEY environment variable."
        )
    
    try:
        agent = get_agent()
        answer = agent.query(request.question)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Set OPENAI_API_KEY environment variable."
        )
    
    try:
        agent = get_agent()
        answer = agent.chat(request.question, request.session_id)
        return ChatResponse(answer=answer, session_id=request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=list[DocumentInfo], tags=["Knowledge Base"])
async def list_documents():
    """List all documents in the knowledge base."""
    try:
        agent = get_agent()
        return agent.get_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/conversation/{session_id}", tags=["Chat"])
async def clear_session(session_id: str):
    """
    Clear conversation history for a specific session.
    
    Use this to reset a user's conversation context.
    After clearing, the next message from this session will start fresh.
    """
    try:
        agent = get_agent()
        success = agent.clear_session(session_id)
        if success:
            return {"message": f"Session '{session_id}' cleared successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found or could not be cleared")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/sessions", tags=["Chat"])
async def list_sessions():
    """List all active session IDs (for debugging/admin purposes)."""
    try:
        agent = get_agent()
        sessions = agent.list_sessions()
        return {"sessions": sessions, "count": len(sessions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
