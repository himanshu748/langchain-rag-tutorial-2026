# LangChain RAG Tutorial (Updated for v1.2.4 - January 2026)

A comprehensive, production-ready RAG (Retrieval Augmented Generation) tutorial using the latest LangChain patterns.

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
pip install langchain langchain-openai langchain-chroma langgraph
export OPENAI_API_KEY="your-key-here"
```

### Option B: Fully Local (No API Key)
```bash
pip install langchain langchain-ollama langchain-chroma langgraph
# Install Ollama: https://ollama.com/download
ollama pull llama3.2
ollama pull nomic-embed-text
```

Then set `USE_OLLAMA = True` in the notebook.

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

- `langchain_rag_tutorial_updated.ipynb` - Main tutorial notebook
- `CHANGELOG.md` - Detailed list of all changes made

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
