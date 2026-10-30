"""Qdrant native sparse retriever (hashed-unigram BM25-like).

Drop-in replacement for SparseRetriever (which uses OpenSearch BM25). Encodes
query via the same hashing trick used at index time in build_pilot_2store.py.

Collection schema (built by scripts/build_pilot_2store.py):
    vectors_config = {"dense": VectorParams(1024, COSINE, on_disk=True)}
    sparse_vectors_config = {"sparse": SparseVectorParams()}

Tokenization + weighting rationale (locked 2026-06-17 M2.4 spike):
  - Regex `[A-Za-z][A-Za-z0-9'+-]*` lowercased, stopwords removed
  - Hashed into 2^18 buckets via md5(tok)[:4] little-endian modulo
  - Weight: log(1 + TF) — matches Lucene BM25 practical similarity approximately
    (no IDF; Qdrant sparse dot product handles ranking).

Spike M2.4 verified on 5K subset: nDCG 0.700, 2× better than OS BM25 0.343.
P3 indexes the full 6.64M corpus for production-realistic A/B in P4.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict

from qdrant_client.http.models import SparseVector

from ..store.qdrant import QdrantStore
from .dense import RetrievalHit

# ─── Sparse encoder (must match scripts/build_pilot_2store.py exactly) ───────

SPARSE_VOCAB_SIZE = 2 ** 18  # 262144 buckets — matches SPLADE vocab

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'+-]*")
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "to", "and", "for", "is", "are", "was",
    "were", "be", "been", "being", "by", "on", "with", "as", "at", "from",
    "this", "that", "these", "those", "it", "its", "we", "our", "i", "you",
    "your", "they", "their", "he", "she", "his", "her", "or", "if", "but",
    "not", "no", "yes", "than", "then", "so", "such", "also", "into",
})


def tokenize_simple(text: str) -> list[str]:
    """Regex tokenizer + stopword filter. Identical to M2.4 spike."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def encode_sparse(text: str) -> tuple[list[int], list[float]]:
    """Hashed-unigram sparse encoding with log(1+TF) weighting.

    Returns (indices, values) sorted by index ascending. Qdrant requires
    ascending indices for SparseVector.

    Matches Lucene BM25 practical similarity approximately (no IDF, no length
    norm — Qdrant sparse dot product handles ranking).
    """
    tokens = tokenize_simple(text)
    if not tokens:
        return [], []
    counts: dict[int, float] = defaultdict(float)
    for tok in tokens:
        h = hashlib.md5(tok.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "little") % SPARSE_VOCAB_SIZE
        counts[idx] += 1.0
    indices = sorted(counts.keys())
    values = [1.0 + math.log(counts[i]) for i in indices]  # log(1+tf)
    return indices, values


# ─── Retriever ───────────────────────────────────────────────────────────────


class QdrantNativeSparseRetriever:
    """Async sparse retriever: query → hashed-unigram → Qdrant sparse search.

    Drop-in replacement for SparseRetriever (BM25 via OpenSearch). Same
    `search()` signature returns list[RetrievalHit] with source='sparse'.
    """

    def __init__(
        self,
        qdrant: QdrantStore,
        top_k: int = 50,
        dense_collection_for_payload: QdrantStore | None = None,
    ):
        """Args:
            qdrant: QdrantStore pointed at the pilot 2-store collection
                (must have sparse vectors configured).
            top_k: default result count.
            dense_collection_for_payload: optional, if you want to fetch
                text/title payload from a separate (e.g. production dense-only)
                collection. Unused in P3 — pilot collection stores the same
                payload as production at index time.
        """
        self.qdrant = qdrant
        self.top_k = top_k

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[RetrievalHit]:
        """Encode query → Qdrant sparse query → RetrievalHit list.

        Filters are translated by build_qdrant_filter (shared util,
        supports scalar/list/range forms — see utils/filters.py).
        """
        from ..utils.filters import build_qdrant_filter

        k = top_k or self.top_k
        sp_indices, sp_values = encode_sparse(query)
        if not sp_indices:
            return []

        flt = build_qdrant_filter(filters)

        res = await self.qdrant.client.query_points(
            collection_name=self.qdrant.collection_name,
            query=SparseVector(indices=sp_indices, values=sp_values),
            using="sparse",
            limit=k,
            query_filter=flt,
            with_payload=True,
            with_vectors=False,
        )

        hits: list[RetrievalHit] = []
        for p in res.points:
            pl = p.payload or {}
            # Qdrant sparse scores are dot products of log(1+TF) — unbounded.
            # Normalize to 0..1 for downstream fusion display (RRF only uses
            # rank so this is cosmetic, same convention as SparseRetriever).
            hits.append(RetrievalHit(
                chunk_id=str(p.id),
                score=float(p.score or 0.0),
                source="sparse",
                text=pl.get("text", ""),
                arxiv_id=pl.get("arxiv_id", ""),
                heading_path=pl.get("heading_path", ""),
                title=pl.get("title", ""),
                chunk_index=pl.get("chunk_index", 0),
                year=pl.get("year"),
                categories=pl.get("categories", []),
                raw_score=float(p.score or 0.0),
                tier=pl.get("tier", "fulltext"),
            ))
        # Normalize after collecting (max across the page)
        max_score = max((h.score for h in hits), default=1.0)
        if max_score > 0:
            for h in hits:
                h.score = h.score / max_score
        return hits
