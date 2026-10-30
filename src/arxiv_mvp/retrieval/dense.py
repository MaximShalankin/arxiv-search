"""Dense retriever: embed query via BGE-M3, search Qdrant."""
from dataclasses import dataclass

from ..ingest.embedder import Embedder
from ..store.qdrant import QdrantStore


@dataclass
class RetrievalHit:
    chunk_id: str
    score: float                # normalized 0..1 (cosine or RRF)
    source: str                 # 'dense' | 'sparse' | 'hybrid'
    text: str
    arxiv_id: str
    heading_path: str
    title: str
    chunk_index: int
    year: int | None
    categories: list[str]
    raw_score: float | None = None  # pre-RRF score (cosine for dense, _score for sparse)
    tier: str = "fulltext"          # ticket-027: "abstract" | "fulltext"


class DenseRetriever:
    """Async dense retriever: query → BGE-M3 vector → Qdrant search."""

    def __init__(
        self,
        embedder: Embedder,
        qdrant: QdrantStore,
        top_k: int = 50,
        vector_name: str | None = None,
    ):
        """vector_name overrides which named vector to query. If None, falls
        back to the QdrantStore's own vector_name (set at construction)."""
        self.embedder = embedder
        self.qdrant = qdrant
        self.top_k = top_k
        self.vector_name = vector_name

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[RetrievalHit]:
        """Embed query and search Qdrant. Returns ranked hits."""
        # Embed (single text, normalized client-side via Embedder contract)
        result = await self.embedder.embed([query])
        if not result.vectors or result.vectors[0] is None:
            return []
        query_vec = result.vectors[0]

        k = top_k or self.top_k
        # DenseRetriever-level override wins; else defer to QdrantStore default.
        vname = self.vector_name if self.vector_name is not None else self.qdrant.vector_name
        points = await self.qdrant.search(query_vec, top_k=k, filters=filters, vector_name=vname)

        hits = []
        for p in points:
            pl = p.payload or {}
            hits.append(RetrievalHit(
                chunk_id=str(p.id),
                score=p.score,
                source="dense",
                text=pl.get("text", ""),
                arxiv_id=pl.get("arxiv_id", ""),
                heading_path=pl.get("heading_path", ""),
                title=pl.get("title", ""),
                chunk_index=pl.get("chunk_index", 0),
                year=pl.get("year"),
                categories=pl.get("categories", []),
                raw_score=p.score,
                tier=pl.get("tier", "fulltext"),
            ))
        return hits
