# LangChain RAG Tutorial API

A FastAPI RAG (Retrieval Augmented Generation) tutorial using LangChain v1-style agents, Chroma, OpenAI embeddings, and optional conversation memory.

## ✨ What's New in This Update

| Feature | Description |
|---------|-------------|
| 🤖 **Agent-Based RAG** | Uses `create_agent` instead of chains - LLM decides when to retrieve |
| 🏠 **Local Model Support** | Ollama integration for free, local inference (no API key needed) |
| 📄 **Advanced PDF Processing** | Tables (PDFPlumber) + Diagrams (GPT-4o Vision) |
| 💬 **Conversation Memory** | Multi-turn chat with `InMemorySaver` |
| 🌊 **Streaming Responses** | Real-time token streaming for better UX |
| 🛠️ **Multi-Tool Agents** | Easily extend with custom tools |

## 🚀 Quick Start

### Option A: With OpenAI API
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill OPENAI_API_KEY in .env
uvicorn main:app --host 127.0.0.1 --port 8000
```

### Option B: Fully Local (No API Key)
```bash
pip install langchain langchain-ollama langchain-chroma langgraph
# Install Ollama: https://ollama.com/download
ollama pull llama3.2
ollama pull nomic-embed-text
```

Then use the notebook's Ollama path for local experimentation.

## 📦 Requirements

```bash
pip install -qU \
    langchain>=1.2.4 \
    langchain-openai \
    langchain-ollama \
    langchain-chroma \
    langchain-community \
    langgraph>=1.0.0 \
    chromadb \
    pypdf \
    pdfplumber \
    pymupdf \
    pillow
```

## Verification

These checks do not require an OpenAI API key:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/langchain-rag-pycache python3 -m compileall main.py rag_agent.py test_session_isolation.py
pytest -q
```

## Runtime Configuration

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Required for `/chat`, `/chat/conversation`, and `/documents` |
| `OPENAI_MODEL` | Chat model name, default `gpt-4o-mini` |
| `DATABASE_URL` | Optional PostgreSQL checkpoint persistence |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins, defaults to localhost only |
| `ENABLE_DEBUG_ENDPOINTS` | Enables `/debug/langsmith` and `/chat/sessions` when `true` |
| `DEBUG_ERRORS` | Returns raw exception details when `true`; keep `false` in production |

## 🔑 Key API Patterns (v1.2.4)

```python
# Agent Creation (NEW - replaces create_retrieval_chain)
from langchain.agents import create_agent
from langchain_core.tools import create_retriever_tool
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model=llm,                    # ChatOpenAI or ChatOllama instance
    tools=[retriever_tool],
    system_prompt="You are...",   # NOT state_modifier!
    checkpointer=InMemorySaver()  # For conversation memory
)

result = agent.invoke({"messages": [HumanMessage(content="question")]})
answer = result["messages"][-1].content
```

## 📊 Why Agents > Chains?

| Chain-Based RAG | Agent-Based RAG |
|-----------------|-----------------|
| Always queries database | LLM decides when to retrieve |
| Wastes tokens on "hi", "thanks" | Answers simple prompts directly |
| Fixed pipeline | Flexible tool selection |
| Hard to extend | Easy to add more tools |

## 📁 Files

- `main.py` - FastAPI app with lazy agent loading
- `rag_agent.py` - LangChain RAG agent implementation
- `test_session_isolation.py` - Local API tests that run without OpenAI credentials
- `langchain_rag_tutorial_updated.ipynb` - Tutorial notebook

## 🧪 Tested With

- Python 3.12
- LangChain 1.2.4
- LangGraph 1.0.6
- OpenAI GPT-4o-mini / GPT-4o
- Ollama llama3.2 / llava

## 📝 License

MIT

---

*Updated January 2026 for LangChain v1.2.4*
