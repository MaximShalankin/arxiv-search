"""Pydantic request/response schemas for the distilled API.

Only the core search/RAG contract is kept. SearchHit/AskResponse stay close
to the original service so existing clients can keep working.
"""
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# Recency preset translated to a {"year": {"$gte": N}} filter.
Recency = Literal["last_12m", "last_36m", "all"]


class HealthResponse(BaseModel):
    status: str = "ok"
    postgres: str
    qdrant: str
    embedder: str
    reranker_loaded: bool
    version: str = "0.1.0"


class SearchHit(BaseModel):
    chunk_id: str
    score: float
    source: str           # 'sparse' | 'dense' | 'hybrid' | 'reranked'
    text: str
    arxiv_id: str
    heading_path: str
    title: str
    chunk_index: int
    year: int | None = None
    categories: list[str] = Field(default_factory=list)
    raw_score: float | None = None
    score_norm: float | None = None  # sigmoid(score) for reranked hits
    tier: str = "fulltext"           # "abstract" | "fulltext"


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    mode: Literal["bm25", "dense", "hybrid", "rerank", "sparse"] = "rerank"
    top_k: int = Field(10, ge=1, le=100)
    top_k_hybrid: int = Field(50, ge=1, le=500)
    filters: dict | None = None
    recency: Recency | None = None
    min_score: float | None = Field(
        None,
        description=(
            "rerank mode only: drop hits whose raw reranker score is below "
            "this value. Ignored for bm25/dense/hybrid."
        ),
    )
    dedupe_by_paper: bool = False

    @field_validator("query")
    @classmethod
    def _query_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty or whitespace-only")
        return v

    @field_validator("mode")
    @classmethod
    def _normalize_mode(cls, v: str) -> str:
        return "bm25" if v == "sparse" else v


class SearchResponse(BaseModel):
    query: str
    mode: str
    top_k: int
    elapsed_ms: float
    hits: list[SearchHit]
    breakdown_ms: dict


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(10, ge=1, le=20)
    recency: Recency | None = None
    min_score: float | None = None
    dedupe_by_paper: bool = False

    @field_validator("query")
    @classmethod
    def _query_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty or whitespace-only")
        return v


class PaperAskRequest(BaseModel):
    """Scoped /ask: answer from ONE paper's chunks only."""
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(10, ge=1, le=20)
    min_score: float | None = None
    dedupe_by_paper: bool = False

    @field_validator("query")
    @classmethod
    def _query_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty or whitespace-only")
        return v


class Citation(BaseModel):
    """Provenance for a claim cited as [N] in an /ask answer."""
    index: int
    chunk_id: str
    arxiv_id: str
    quote: str
    section: str
    abs_url: str = ""
    pdf_url: str = ""


class Grounding(BaseModel):
    """Heuristic confidence for an /ask answer.

    ``context_strength`` measures retrieval relevance of cited passages, not
    answer truth. Thresholds are uncalibrated and should be treated as a
    warning signal only.
    """
    confidence: Literal["high", "medium", "low", "refusal"]
    context_strength: float
    n_cited: int
    n_context: int
    refusal: bool
    warnings: list[str] = Field(default_factory=list)


class AskResponse(BaseModel):
    query: str
    answer: str
    context: list[SearchHit]
    citations: list[Citation] = Field(default_factory=list)
    elapsed_ms: float
    grounding: Grounding


class PaperResponse(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    categories: list[str]
    abstract: str
    n_chunks: int
    chunk_ids: list[str]
    tier: str = "fulltext"


class CorpusStatsResponse(BaseModel):
    n_papers: int
    n_chunks: int
    year_range: list[int | None]
    top_categories: list[dict]
