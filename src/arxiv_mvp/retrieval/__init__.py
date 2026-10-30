"""Retrieval layer: dense, native sparse, RRF fusion, cross-encoder rerank."""
from .dense import DenseRetriever, RetrievalHit
from .fusion import hybrid_search, rrf_fuse
from .qdrant_native_sparse import (
    SPARSE_VOCAB_SIZE,
    QdrantNativeSparseRetriever,
    encode_sparse,
)
from .reranker import Reranker

__all__ = [
    "SPARSE_VOCAB_SIZE",
    "DenseRetriever",
    "QdrantNativeSparseRetriever",
    "Reranker",
    "RetrievalHit",
    "encode_sparse",
    "hybrid_search",
    "rrf_fuse",
]
