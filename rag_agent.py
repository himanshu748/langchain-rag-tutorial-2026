"""
RAG Agent Module - Extracted from LangChain RAG Tutorial
LangChain v1.2.4 (January 2026)
"""

import os

# LangSmith Tracing - auto-enabled when LANGSMITH_TRACING=true
if os.getenv("LANGSMITH_TRACING"):
    print("✅ LangSmith tracing enabled - view traces at https://smith.langchain.com")
from typing import Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import InMemorySaver
import psycopg


# Sample documents for demonstration
SAMPLE_DOCUMENTS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models. It provides tools for prompt management, chains, and agents.",
        metadata={"source": "langchain_intro.txt", "page": 1}
    ),
    Document(
        page_content="RAG (Retrieval Augmented Generation) combines retrieval and generation to produce more accurate and up-to-date responses. It works by retrieving relevant documents from a knowledge base.",
        metadata={"source": "rag_overview.txt", "page": 1}
    ),
    Document(
        page_content="Vector databases store data as high-dimensional vectors, enabling similarity search. Popular options include Chroma, Pinecone, and Weaviate.",
        metadata={"source": "vector_db.txt", "page": 1}
    ),
    Document(
        page_content="Embeddings are numerical representations of text that capture semantic meaning. OpenAI embeddings and sentence-transformers are commonly used.",
        metadata={"source": "embeddings.txt", "page": 1}
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications. It powers LangChain's agent framework with features like persistence and streaming.",
        metadata={"source": "langgraph.txt", "page": 1}
    ),
    Document(
        page_content="The create_agent function from langchain.agents is the modern way to build agents in LangChain v1.2+. It provides a simple interface with system prompts and tool integration.",
        metadata={"source": "create_agent.txt", "page": 1}
    ),
    Document(
        page_content="InMemorySaver from langgraph.checkpoint.memory enables conversation persistence. Each thread_id maintains separate conversation history for multi-user support.",
        metadata={"source": "memory.txt", "page": 1}
    ),
]

RAG_SYSTEM_PROMPT = """You are a helpful AI assistant with access to a knowledge base about LangChain, RAG, and related technologies.

When answering questions:
1. ALWAYS use the knowledge_base_search tool to find relevant information
2. Base your answers on the retrieved documents
3. If the information isn't in the knowledge base, say so honestly
4. Cite the source when providing information

Be concise but thorough in your responses."""


class RAGAgent:
    """RAG Agent with conversation memory support."""
    
    def __init__(self):
        """Initialize the RAG agent with vector store and tools."""
        # Initialize LLM
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Initialize embeddings and vector store
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(
            documents=SAMPLE_DOCUMENTS,
            embedding=self.embeddings,
        )
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        # Create retriever tool
        self.retriever_tool = create_retriever_tool(
            self.retriever,
            name="knowledge_base_search",
            description="""Search the knowledge base for information about LangChain, 
            RAG, vector databases, embeddings, and agents. Use this tool when you need 
            to find specific information from the documents."""
        )
        
        # PostgreSQL checkpointer for persistent conversations
        # Each session_id (thread_id) has completely isolated conversation history
        self.db_uri = os.getenv("DATABASE_URL")
        self.checkpointer = None
        self.pg_conn = None
        
        if self.db_uri:
            try:
                # Render uses postgres:// but psycopg needs postgresql://
                db_uri = self.db_uri.replace("postgres://", "postgresql://")
                self.pg_conn = psycopg.connect(db_uri, autocommit=True)
                self.checkpointer = PostgresSaver(self.pg_conn)
                self.checkpointer.setup()  # Create tables if they don't exist
                print("✅ PostgreSQL checkpointer initialized")
            except Exception as e:
                print(f"⚠️ PostgreSQL connection failed: {e}")
                print("   Falling back to InMemorySaver")
                self.checkpointer = InMemorySaver()
        else:
            print("⚠️ DATABASE_URL not set - using InMemorySaver (no persistence)")
            self.checkpointer = InMemorySaver()
        
        # Create simple agent (no memory - for single queries)
        self.simple_agent = create_agent(
            model=self.llm,
            tools=[self.retriever_tool],
            system_prompt=RAG_SYSTEM_PROMPT,
        )
        
        # Create conversational agent (with memory)
        self.conversational_agent = create_agent(
            model=self.llm,
            tools=[self.retriever_tool],
            system_prompt=RAG_SYSTEM_PROMPT,
            checkpointer=self.checkpointer,
        )
    
    def query(self, question: str) -> str:
        """Single-turn query without memory."""
        try:
            response = self.simple_agent.invoke({
                "messages": [HumanMessage(content=question)]
            })
            return response["messages"][-1].content
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def chat(self, question: str, session_id: str = "default") -> str:
        """Multi-turn conversation with memory."""
        try:
            config = {"configurable": {"thread_id": session_id}}
            response = self.conversational_agent.invoke(
                {"messages": [HumanMessage(content=question)]},
                config=config
            )
            return response["messages"][-1].content
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def get_documents(self) -> list[dict]:
        """Return list of documents in the knowledge base."""
        return [
            {
                "content": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 1)
            }
            for doc in SAMPLE_DOCUMENTS
        ]

    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a specific session."""
        if not self.pg_conn:
            return False
        try:
            cursor = self.pg_conn.cursor()
            cursor.execute(
                "DELETE FROM checkpoints WHERE thread_id = %s",
                (session_id,)
            )
            cursor.execute(
                "DELETE FROM checkpoint_writes WHERE thread_id = %s",
                (session_id,)
            )
            return True
        except Exception:
            return False

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        if not self.pg_conn:
            return []
        try:
            cursor = self.pg_conn.cursor()
            cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
            return [row[0] for row in cursor.fetchall()]
        except Exception:
            return []


# Singleton instance
_agent_instance: Optional[RAGAgent] = None


def get_agent() -> RAGAgent:
    """Get or create the RAG agent instance."""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = RAGAgent()
    return _agent_instance
