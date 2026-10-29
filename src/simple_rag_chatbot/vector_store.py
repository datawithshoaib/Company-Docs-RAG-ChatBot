"""
ChromaDB helpers.

NOTE FOR WORKSHOP FACILITATORS: the exported Colab script this project
was rebuilt from called four functions that were never defined in the
exported file (``build_multivector_indexes``, ``resolve_source_file``,
``_query_vector_store``, ``_rrf``) — they must have lived in Colab
cells that were not included in the download. The implementations
below are original, written to match how they are used elsewhere in
the codebase (see ``multivector_retrieve`` and ``graph.py``). If you
have the original cells, feel free to swap these out; they are
intentionally kept in one small, well-documented module so that's easy.
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    import chromadb


# ============================================================
# BASIC COLLECTION INSPECTION
# ============================================================

def list_documents(collection) -> list:
    """Return unique source files from ChromaDB."""

    all_items = collection.get(include=["metadatas"])

    sources = sorted({
        m.get("source_file", "unknown")
        for m in all_items["metadatas"]
        if m
    })

    return sources


def get_available_metadata(collection) -> dict:
    """
    Inspect metadata fields and their available values
    from the Chroma collection.
    """

    all_items = collection.get(include=["metadatas"])

    metadata_values: dict[str, set] = {}

    for meta in all_items.get("metadatas", []):
        if not meta:
            continue

        for key, value in meta.items():
            if value is None:
                continue

            metadata_values.setdefault(key, set()).add(str(value))

    return {key: sorted(values) for key, values in metadata_values.items()}


def resolve_source_file(requested_value: str, all_sources: list[str]) -> str | None:
    """
    Resolve a (possibly partial, possibly fuzzy) filename mentioned by
    the user/LLM against the real source filenames stored in Chroma.

    Resolution order:
      1. Exact match (case-insensitive).
      2. Unique case-insensitive substring match.
      3. Best fuzzy match above a similarity threshold (only if it is
         unambiguous).
      4. Otherwise, give up and return None rather than guess.
    """

    if not requested_value or not all_sources:
        return None

    requested_lower = requested_value.strip().lower()

    # 1. Exact match.
    for source in all_sources:
        if source.lower() == requested_lower:
            return source

    # 2. Unique substring match (either direction).
    substring_matches = [
        source
        for source in all_sources
        if requested_lower in source.lower() or source.lower() in requested_lower
    ]

    if len(substring_matches) == 1:
        return substring_matches[0]

    # 3. Fuzzy match, only trust it if there is one clear winner.
    close = difflib.get_close_matches(
        requested_value, all_sources, n=2, cutoff=0.6
    )

    if len(close) == 1:
        return close[0]

    return None


# ============================================================
# MULTI-VECTOR INDEXES (summary vector + entity/topic vector)
# ============================================================

def _document_metadata_text(meta: dict) -> str:
    """Compact text representation of a chunk's document-level metadata,
    used to build the "entity/topic" vector index."""

    parts = [
        meta.get("document_type", ""),
        meta.get("document_category", ""),
        meta.get("location", ""),
        f"entities: {meta.get('entities', '')}",
        f"topics: {meta.get('topics', '')}",
    ]
    return " | ".join(p for p in parts if p)


def build_multivector_indexes(
    client: "chromadb.ClientAPI",
    collection,
    embedder,
    summary_collection_name: str,
    metadata_collection_name: str,
    batch_size: int = 64,
):
    """
    Build (or refresh) two auxiliary Chroma collections derived from
    the document-level metadata already stored on each chunk of
    ``collection``:

    - ``summary_collection_name``: one vector per chunk, embedding that
       chunk's *document-level* one-line summary. This lets a query
       match a document by what it is about, not just by literal
       wording in the chunk text.
    - ``metadata_collection_name``: one vector per chunk, embedding a
       compact string of document type / category / location /
       entities / topics. This lets a query match a document by the
       things it mentions (e.g. "Bangalore", "maternity leave").

    Both collections use the *same chunk ids* as ``collection`` so
    fused results can be traced back to the original chunk.

    Returns (summary_collection, metadata_collection).
    """

    all_items = collection.get(include=["metadatas"])
    ids = all_items.get("ids", [])
    metadatas = all_items.get("metadatas", [])

    summary_collection = client.get_or_create_collection(summary_collection_name)
    metadata_collection = client.get_or_create_collection(metadata_collection_name)

    if not ids:
        return summary_collection, metadata_collection

    summary_texts = [
        (meta or {}).get("summary", "") or (meta or {}).get("source_file", "")
        for meta in metadatas
    ]
    metadata_texts = [_document_metadata_text(meta or {}) for meta in metadatas]

    for start in range(0, len(ids), batch_size):
        end = start + batch_size

        batch_ids = ids[start:end]
        batch_metas = metadatas[start:end]

        batch_summary_texts = summary_texts[start:end]
        batch_metadata_texts = metadata_texts[start:end]

        summary_embeddings = embedder.encode(batch_summary_texts).tolist()
        metadata_embeddings = embedder.encode(batch_metadata_texts).tolist()

        summary_collection.upsert(
            ids=batch_ids,
            embeddings=summary_embeddings,
            documents=batch_summary_texts,
            metadatas=batch_metas,
        )

        metadata_collection.upsert(
            ids=batch_ids,
            embeddings=metadata_embeddings,
            documents=batch_metadata_texts,
            metadatas=batch_metas,
        )

    return summary_collection, metadata_collection


# ============================================================
# QUERYING + RECIPROCAL RANK FUSION
# ============================================================

def _query_vector_store(
    store,
    query_embedding,
    candidate_k: int,
    where: dict | None = None,
) -> list[dict]:
    """
    Query a single Chroma collection and return a flat list of
    candidates: ``[{"id", "document", "metadata", "distance"}, ...]``
    ordered by relevance (Chroma already returns results ordered by
    distance).
    """

    if query_embedding and isinstance(query_embedding[0], (int, float)):
        query_embeddings = [query_embedding]
    else:
        query_embeddings = query_embedding

    query_kwargs: dict[str, Any] = {
        "query_embeddings": query_embeddings,
        "n_results": candidate_k,
        "include": ["documents", "metadatas", "distances"],
    }

    if where:
        query_kwargs["where"] = where

    try:
        results = store.query(**query_kwargs)
    except Exception:
        return []

    ids = (results.get("ids") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances_raw = results.get("distances")
    distances = (
        distances_raw[0]
        if distances_raw and len(distances_raw) > 0 and distances_raw[0]
        else [None] * len(ids)
    )

    return [
        {"id": _id, "document": doc, "metadata": meta or {}, "distance": dist}
        for _id, doc, meta, dist in zip(ids, documents, metadatas, distances)
    ]


def _rrf(
    result_lists: Iterable[list[dict]],
    candidate_k: int,
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion across several ranked candidate lists.

    Each candidate contributes ``1 / (k + rank)`` to its fused score
    for every list it appears in (rank is 0-indexed position within
    that list). Candidates are deduplicated by id; the first-seen
    document/metadata is kept.

    This is the standard RRF formula (Cormack et al., 2009), and is a
    simple, parameter-light way to combine rankings from semantically
    different vector spaces (content, summary, metadata) without
    needing to normalize distances across them.
    """

    scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)

            if item_id not in items:
                items[item_id] = item

    fused = sorted(
        items.values(),
        key=lambda item: scores[item["id"]],
        reverse=True,
    )

    return fused[:candidate_k]


def multivector_retrieve(
    content,
    summary,
    metadata,
    query_embedding,
    candidate_k: int,
    where: dict | None = None,
):
    """
    Retrieve from the content, summary, and entity/topic vector spaces
    independently, then fuse the three ranked lists with Reciprocal
    Rank Fusion.

    Returns (fused, content_results, summary_results, metadata_results).
    """

    content_r = _query_vector_store(content, query_embedding, candidate_k, where)
    summary_r = _query_vector_store(summary, query_embedding, candidate_k, where)
    metadata_r = _query_vector_store(metadata, query_embedding, candidate_k, where)

    fused = _rrf([content_r, summary_r, metadata_r], candidate_k)

    return fused, content_r, summary_r, metadata_r
