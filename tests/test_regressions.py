"""Core regression tests.

- arxiv_id -> year keeps the century ([7, 26] bug).
- Embedder.embed_many batches transparently.
- Sparse encoder is deterministic and index-sorted.
- The API exposes only the distilled core routes.
"""
import asyncio

from arxiv_mvp.api.app import _arxiv_id_to_year, app
from arxiv_mvp.ingest.chunker import _split_long_unit
from arxiv_mvp.ingest.embedder import Embedder
from arxiv_mvp.retrieval.qdrant_native_sparse import (
    SPARSE_VOCAB_SIZE,
    encode_sparse,
)


# ─── arxiv_id -> year ───────────────────────────────────────────────────────
def test_year_old_style():
    assert _arxiv_id_to_year("0701.1234") == 2007


def test_year_new_style():
    assert _arxiv_id_to_year("2606.12345") == 2026


def test_year_edge_minimum():
    assert _arxiv_id_to_year("9107.0001") == 1991


# ─── embed_many batching ───────────────────────────────────────────────────
def test_embed_many_batches_transparently(monkeypatch):
    """40 texts with batch_size=16 -> 3 POSTs (16/16/8), no ValueError."""
    calls = []

    async def fake_post(self, texts):
        calls.append(len(texts))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(Embedder, "_post_once", fake_post)
    emb = Embedder(batch_size=16, timeout_s=600.0)
    out = asyncio.run(emb.embed_many(["chunk"] * 40))

    assert len(out) == 40
    assert all(v is not None for v in out)
    assert calls == [16, 16, 8]


# ─── sparse encoder ────────────────────────────────────────────────────────
def test_sparse_encoder_is_deterministic():
    a = encode_sparse("retrieval augmented generation with retrieval")
    b = encode_sparse("retrieval augmented generation with retrieval")
    assert a == b
    indices, values = a
    assert indices == sorted(indices)
    assert all(0 <= i < SPARSE_VOCAB_SIZE for i in indices)
    assert len(indices) == len(values)


def test_sparse_encoder_empty_text():
    assert encode_sparse("") == ([], [])


# ─── oversized table split ─────────────────────────────────────────────────
def test_oversized_table_row_does_not_crash():
    """A single row larger than the hard cap exercises the char-window split."""
    header = "| a | b |\n|---|---|"
    row = "| x | " + ("y" * 12000) + " |"
    units = _split_long_unit("Table 1", "table", header + "\n" + row)
    assert units
    assert all(u[1] == "table" for u in units)
    assert all(u[2].startswith("| a | b |") for u in units)


# ─── route surface ─────────────────────────────────────────────────────────
def test_core_routes_are_present():
    paths = {r.path for r in app.routes}
    for path in (
        "/",
        "/health",
        "/stats",
        "/search",
        "/ask",
        "/paper/{arxiv_id}",
        "/paper/{arxiv_id}/ask",
        "/warmup",
    ):
        assert path in paths


def test_non_core_routes_are_absent():
    paths = {r.path for r in app.routes}
    for path in (
        "/seo/analyze",
        "/claim-matrix",
        "/conflicting-evidence",
        "/related/{arxiv_id}",
        "/similar/{arxiv_id}",
        "/export",
        "/eval/prompt",
        "/session/{session_id}",
        "/promote/{arxiv_id}",
        "/paper/{arxiv_id}/tables",
    ):
        assert path not in paths
