"""RRF fusion + hybrid search orchestrator.

RRF (Reciprocal Rank Fusion):
    score(d) = Σ_i  w_i / (k + rank_i(d))
where:
    d = chunk_id
    k = 60 (standard, ticket-002 M3 contract)
    rank_i(d) = 1-based position of d in ranking i
    w_i = weight of retrieval method i (default 0.5/0.5 for [dense, sparse])

References: Cormack et al. 2009; ticket-002.1 M1.3.
"""
import asyncio
from collections import defaultdict

from .dense import DenseRetriever, RetrievalHit
from .qdrant_native_sparse import QdrantNativeSparseRetriever


def rrf_fuse(
    dense_hits: list[RetrievalHit],
    sparse_hits: list[RetrievalHit],
    k: int = 60,
    weights: tuple[float, float] = (0.5, 0.5),
    top_k: int | None = None,
) -> list[RetrievalHit]:
    """Fuse dense + sparse rankings via RRF. Returns sorted hybrid hits."""
    scores: dict[str, float] = defaultdict(float)
    payloads: dict[str, RetrievalHit] = {}

    # Dense ranking
    w_dense = weights[0] / sum(weights)
    for rank, hit in enumerate(dense_hits, start=1):
        cid = hit.chunk_id
        scores[cid] += w_dense / (k + rank)
        payloads[cid] = hit

    # Sparse ranking
    w_sparse = weights[1] / sum(weights)
    for rank, hit in enumerate(sparse_hits, start=1):
        cid = hit.chunk_id
        scores[cid] += w_sparse / (k + rank)
        if cid not in payloads:
            payloads[cid] = hit

    # Sort by fused score desc
    sorted_ids = sorted(scores.items(), key=lambda x: -x[1])

    if top_k is not None:
        sorted_ids = sorted_ids[:top_k]

    # Build hybrid hits
    fused_hits = []
    for cid, fused_score in sorted_ids:
        base = payloads[cid]
        fused_hits.append(RetrievalHit(
            chunk_id=cid,
            score=fused_score,
            source="hybrid",
            text=base.text,
            arxiv_id=base.arxiv_id,
            heading_path=base.heading_path,
            title=base.title,
            chunk_index=base.chunk_index,
            year=base.year,
            categories=base.categories,
            raw_score=base.raw_score,  # from whichever retriever found it first
            tier=base.tier,
        ))
    return fused_hits


async def hybrid_search(
    query: str,
    dense_retriever: DenseRetriever,
    sparse_retriever: QdrantNativeSparseRetriever,
    top_k_per_retriever: int = 50,
    top_k_final: int = 10,
    rrf_k: int = 60,
    weights: tuple[float, float] = (0.5, 0.5),
    filters: dict | None = None,
) -> list[RetrievalHit]:
    """Parallel dense + sparse search, then RRF fusion."""
    dense_hits, sparse_hits = await asyncio.gather(
        dense_retriever.search(query, top_k=top_k_per_retriever, filters=filters),
        sparse_retriever.search(query, top_k=top_k_per_retriever, filters=filters),
    )

    return rrf_fuse(
        dense_hits, sparse_hits,
        k=rrf_k, weights=weights, top_k=top_k_final,
    )
