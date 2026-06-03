# Repository Instructions

## Scope
This repository is a LangChain/FastAPI RAG tutorial and demo API. Keep it safe to clone, test, and deploy without exposing secrets or requiring paid API calls during basic verification.

## Commands
- `PYTHONPYCACHEPREFIX=/private/tmp/langchain-rag-pycache python3 -m compileall main.py rag_agent.py test_session_isolation.py`
- `pytest -q`
- `uvicorn main:app --host 127.0.0.1 --port 8000`

## Conventions
- Do not commit bytecode caches, vector-store directories, `.env`, API keys, LangSmith keys, database URLs, or notebook checkpoint files.
- Keep FastAPI import and health checks working without LangChain packages or `OPENAI_API_KEY`; heavy RAG imports should stay lazy.
- Gate debug/admin endpoints behind `ENABLE_DEBUG_ENDPOINTS=true`.
- Keep CORS explicit via `ALLOWED_ORIGINS`; avoid wildcard origins for deployed services.
- Prefer local API tests over tests that hit Render or other external deployments.
