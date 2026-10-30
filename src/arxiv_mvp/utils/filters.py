"""Translate API-level filter dict → Qdrant Filter object.

Shared by QdrantStore.search (dense) and QdrantNativeSparseRetriever.search
(sparse) — previously duplicated in both modules (bug-001 A4).

Supported syntax (Mongo/Qdrant hybrid):
    {"year": 2024}                          → MatchValue (exact)
    {"categories": ["cs.LG", "cs.CL"]}      → MatchAny (membership)
    {"year": {"$gte": 2022, "$lte": 2024}}  → Range (inclusive)
    {"category": "cs.LG"}                   → alias: same as {"categories": [...]}

Unknown operators (e.g. "$bogus") raise ValueError → API route returns 400.

Field aliases (bug-001 A5): singular → plural for user-friendliness.
"""
from __future__ import annotations

from qdrant_client.http import models

# Singular → plural / canonical name. Add more if users hit wrong names.
_FIELD_ALIASES = {
    "category": "categories",
}

# Supported range operators → Qdrant Range kwarg names.
_RANGE_OPS = {"$gt": "gt", "$gte": "gte", "$lt": "lt", "$lte": "lte"}


class FilterError(ValueError):
    """Raised on unsupported filter syntax. Mapped to HTTP 400 by caller."""


def build_qdrant_filter(filters: dict | None) -> models.Filter | None:
    """Build a Qdrant Filter from API dict. Returns None if filters is None/empty.

    Raises FilterError on malformed input (unknown operator, empty range dict,
    non-scalar/list/dict values).
    """
    if not filters:
        return None

    must: list[models.FieldCondition] = []
    for raw_key, value in filters.items():
        key = _FIELD_ALIASES.get(raw_key, raw_key)

        if isinstance(value, dict):
            # Range form: {"$gte": N, "$lte": M}
            if not value:
                raise FilterError(
                    f"filter '{raw_key}': empty range dict — use scalar or "
                    f"{{$gte: N}} / {{$lte: M}} form"
                )
            unknown = [op for op in value if op not in _RANGE_OPS]
            if unknown:
                raise FilterError(
                    f"filter '{raw_key}': unknown operator(s) {unknown}. "
                    f"Supported: {sorted(_RANGE_OPS)}"
                )
            range_kwargs = {
                _RANGE_OPS[op]: val for op, val in value.items()
            }
            must.append(
                models.FieldCondition(
                    key=key,
                    range=models.Range(**range_kwargs),
                )
            )
        elif isinstance(value, list):
            if not value:
                raise FilterError(
                    f"filter '{raw_key}': empty list — drop key or provide values"
                )
            must.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchAny(any=value),
                )
            )
        elif isinstance(value, (str, int, float, bool)):
            must.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
            )
        else:
            raise FilterError(
                f"filter '{raw_key}': unsupported value type "
                f"{type(value).__name__} — use scalar, list, or "
                f"{{$gte: N, $lte: M}} dict"
            )

    if not must:
        return None
    return models.Filter(must=must)


# bug-001 P4 (2026-06-24): recency preset → {"year": {"$gte": N}}.
# Approximate (whole-year granularity; corpus indexes year, not month).
# last_12m from June 2026 ≈ year >= 2025; last_36m ≈ year >= 2023.
def apply_recency(filters: dict | None, recency: str | None) -> dict | None:
    """Merge a recency preset into a filter dict. Returns the merged dict.

    `recency`: 'last_12m' | 'last_36m' | 'all' | None.
    Adds/merges {"year": {"$gte": N}} (N = current_year − window). Preserves any
    user-supplied year range and other keys. Raises FilterError on unknown
    preset (caller maps to HTTP 400).
    """
    if recency in (None, "all"):
        return filters
    import datetime
    cur_year = datetime.datetime.now().year
    if recency == "last_12m":
        gte = cur_year - 1
    elif recency == "last_36m":
        gte = cur_year - 3
    else:
        raise FilterError(
            f"unknown recency '{recency}': use last_12m|last_36m|all"
        )
    out = dict(filters) if filters else {}
    year = dict(out.get("year", {}))
    if isinstance(year, dict):
        year["$gte"] = gte
    else:
        # user gave {"year": 2024} (exact) — override with range gte.
        year = {"$gte": gte}
    out["year"] = year
    return out
