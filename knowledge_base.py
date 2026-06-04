"""Lightweight tutorial corpus used by both the API and LangChain agent."""

from typing import TypedDict


class KnowledgeBaseDocument(TypedDict):
    content: str
    source: str
    page: int


SAMPLE_KNOWLEDGE_BASE: tuple[KnowledgeBaseDocument, ...] = (
    {
        "content": (
            "LangChain is a framework for developing applications powered by "
            "language models. It provides tools for prompt management, chains, and agents."
        ),
        "source": "langchain_intro.txt",
        "page": 1,
    },
    {
        "content": (
            "RAG (Retrieval Augmented Generation) combines retrieval and generation "
            "to produce more accurate and up-to-date responses. It works by retrieving "
            "relevant documents from a knowledge base."
        ),
        "source": "rag_overview.txt",
        "page": 1,
    },
    {
        "content": (
            "Vector databases store data as high-dimensional vectors, enabling "
            "similarity search. Popular options include Chroma, Pinecone, and Weaviate."
        ),
        "source": "vector_db.txt",
        "page": 1,
    },
    {
        "content": (
            "Embeddings are numerical representations of text that capture semantic "
            "meaning. OpenAI embeddings and sentence-transformers are commonly used."
        ),
        "source": "embeddings.txt",
        "page": 1,
    },
    {
        "content": (
            "LangGraph is a library for building stateful, multi-actor applications. "
            "It powers LangChain's agent framework with features like persistence and streaming."
        ),
        "source": "langgraph.txt",
        "page": 1,
    },
    {
        "content": (
            "The create_agent function from langchain.agents is the modern way to build "
            "agents in LangChain v1.2+. It provides a simple interface with system prompts "
            "and tool integration."
        ),
        "source": "create_agent.txt",
        "page": 1,
    },
    {
        "content": (
            "InMemorySaver from langgraph.checkpoint.memory enables conversation persistence. "
            "Each thread_id maintains separate conversation history for multi-user support."
        ),
        "source": "memory.txt",
        "page": 1,
    },
)


def list_sample_documents(*, truncate: bool = False) -> list[KnowledgeBaseDocument]:
    """Return serializable corpus metadata without importing LangChain."""
    documents: list[KnowledgeBaseDocument] = []
    for document in SAMPLE_KNOWLEDGE_BASE:
        content = document["content"]
        if truncate and len(content) > 100:
            content = content[:100] + "..."
        documents.append(
            {
                "content": content,
                "source": document["source"],
                "page": document["page"],
            }
        )
    return documents
