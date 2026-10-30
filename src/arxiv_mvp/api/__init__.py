"""FastAPI service for the distilled arxiv-search core.

Endpoints:
  GET  /health                 liveness + storage connectivity
  GET  /stats                  corpus counts (cached)
  POST /search                 sparse | dense | hybrid | rerank
  POST /ask                    retrieve -> rerank -> LLM with [N] citations
  GET  /paper/{arxiv_id}       paper metadata + chunk ids
  POST /paper/{arxiv_id}/ask   scoped RAG for one paper
  POST /warmup                 preload the reranker
"""
from .app import app

__all__ = ["app"]
