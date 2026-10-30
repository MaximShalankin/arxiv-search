"""Qdrant async store for the two-store index (dense + native sparse).

Collection config:
  - named vectors: ``dense`` (BGE-M3, 1024-d, cosine) and ``sparse``
    (hashed-unigram, see ``retrieval/qdrant_native_sparse.py``);
  - scalar INT8 quantization on disk, rescore against fp32 originals;
  - HNSW on disk; payload carries metadata and chunk provenance.
"""
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SearchParams,
    SparseVectorParams,
    VectorParams,
)

from ..utils.logging import get_logger

log = get_logger("qdrant")

DEFAULT_COLLECTION = "arxiv_chunks"
VECTOR_SIZE = 1024  # BGE-M3 dense


class QdrantStore:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = DEFAULT_COLLECTION,
        timeout_s: float = 60.0,
        vector_name: str | None = None,
        vector_size: int = VECTOR_SIZE,
    ):
        """vector_name is the dense vector name for search (``dense`` in this
        project). vector_size is fixed by the embedding model; BGE-M3 = 1024."""
        self.url = url
        self.collection_name = collection_name
        self.timeout_s = timeout_s
        self.vector_name = vector_name
        self.vector_size = vector_size
        self._client: AsyncQdrantClient | None = None

    async def __aenter__(self) -> "QdrantStore":
        # Default 5s is too tight when a parallel writer (e.g. ingestion
        # pipeline) holds collection-level locks during batch upserts. 60s
        # gives breathing room for both readers and writers to coexist.
        self._client = AsyncQdrantClient(url=self.url, timeout=self.timeout_s)
        return self

    async def __aexit__(self, *a):
        if self._client is not None:
            await self._client.close()
            self._client = None

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            raise RuntimeError("QdrantStore not entered (use 'async with')")
        return self._client

    async def init_collection(self, recreate: bool = False):
        """Create the hybrid collection (named ``dense`` + ``sparse``).

        Idempotent: an existing collection is left untouched. ``recreate=True``
        deletes it first — destructive, use only for a deliberate rebuild.
        """
        if recreate:
            await self.client.delete_collection(self.collection_name)
            log.info("collection_deleted", name=self.collection_name)

        exists = await self.client.collection_exists(self.collection_name)
        if exists:
            log.info("collection_exists", name=self.collection_name)
            return

        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                    on_disk=True,
                ),
            },
            sparse_vectors_config={"sparse": SparseVectorParams()},
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                ),
            ),
            hnsw_config=HnswConfigDiff(on_disk=True),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=50000,
            ),
        )
        log.info("collection_created",
                 name=self.collection_name, vector_size=self.vector_size,
                 vectors=["dense", "sparse"], quantization="INT8", rescore=True)

    async def upsert_chunks(
        self,
        items: list[dict],  # each: {point_id, vector, payload}
    ):
        """Bulk upsert. Items must include normalized vectors."""
        points = [
            models.PointStruct(
                id=item["point_id"],
                vector=item["vector"],
                payload=item["payload"],
            )
            for item in items
        ]
        await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 50,
        filters: dict | None = None,
        vector_name: str | None = None,
    ):
        """Dense search with rescore=True (re-rank against fp32 originals).

        vector_name overrides the store-level self.vector_name for this call.
        Filters dict is translated by build_qdrant_filter (shared util,
        supports scalar/list/range forms — see utils/filters.py).
        """
        from ..utils.filters import build_qdrant_filter

        flt = build_qdrant_filter(filters)

        using = vector_name if vector_name is not None else self.vector_name
        res = await self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=using,
            limit=top_k,
            query_filter=flt,
            search_params=SearchParams(
                quantization=models.QuantizationSearchParams(
                    rescore=True,  # re-score top-K against fp32 on disk
                    ignore=False,
                )
            ),
            with_payload=True,
            with_vectors=False,
        )
        return res.points
