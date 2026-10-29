"""Generate (and cache) document-level metadata — summary, entities,
document type/category, location, topics — for every source document
in the collection, using an LLM chain."""

from __future__ import annotations

import hashlib
import re


def _metadata_fingerprint(source, texts) -> str:
    """Stable fingerprint for source + all chunk content."""
    payload = str(source or "") + "\n" + "\n".join(str(t or "") for t in texts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _metadata_is_complete(meta) -> bool:
    """Return True when all metadata required by retrieval exists."""
    required_fields = [
        "summary",
        "entities",
        "document_type",
        "document_category",
        "location",
        "topics",
    ]
    return all(str(meta.get(field, "") or "").strip() for field in required_fields)


def infer_location_from_source(source_file) -> str:
    """
    Conservative filename fallback.

    Example:
        Holiday Calendar 2025 - Bangalore.pdf
        -> Bangalore
    """
    if not source_file:
        return ""

    filename = str(source_file).rsplit("/", 1)[-1].strip()

    match = re.search(
        r"-\s*([A-Za-z][A-Za-z .&'/-]*)\.pdf$", filename, re.IGNORECASE
    )

    return match.group(1).strip() if match else ""


def _normalise_metadata_list(values) -> str:
    if not values:
        return ""
    cleaned = []
    for value in values:
        value = str(value).strip()
        if value and value not in cleaned:
            cleaned.append(value)
    return ", ".join(cleaned)


def enrich_collection_with_document_metadata(
    collection,
    document_metadata_chain,
    max_chars_per_document: int = 12000,
) -> dict:
    """
    Incrementally generate and persist document-level metadata.

    Existing complete metadata is reused. The LLM is called only when
    metadata is missing/incomplete. A content fingerprint is stored on
    every chunk so future runs can detect content changes.
    """

    all_items = collection.get(include=["documents", "metadatas"])

    ids = all_items.get("ids", [])
    documents = all_items.get("documents", [])
    metadatas = all_items.get("metadatas", [])

    if not ids:
        return {}

    grouped = {}

    for item_id, doc, meta in zip(ids, documents, metadatas):
        meta = meta or {}
        source = meta.get("source_file", "unknown")

        grouped.setdefault(source, {"ids": [], "texts": [], "metadatas": []})

        grouped[source]["ids"].append(item_id)
        grouped[source]["texts"].append(doc or "")
        grouped[source]["metadatas"].append(meta)

    generated = {}
    reused_count = 0
    generated_count = 0

    print("\nChecking document-level metadata...")

    for source, data in grouped.items():
        fingerprint = _metadata_fingerprint(source, data["texts"])

        cached = None

        # First preference: exact fingerprint + complete metadata.
        for meta in data["metadatas"]:
            if (
                str(meta.get("metadata_fingerprint", "") or "").strip() == fingerprint
                and _metadata_is_complete(meta)
            ):
                cached = {
                    "summary": str(meta.get("summary", "") or "").strip(),
                    "entities": str(meta.get("entities", "") or "").strip(),
                    "document_type": str(meta.get("document_type", "") or "").strip(),
                    "document_category": str(
                        meta.get("document_category", "") or ""
                    ).strip(),
                    "location": str(meta.get("location", "") or "").strip(),
                    "topics": str(meta.get("topics", "") or "").strip(),
                }
                break

        # Backward compatibility: reuse existing complete metadata
        # created before fingerprints were introduced.
        if cached is None:
            for meta in data["metadatas"]:
                if _metadata_is_complete(meta):
                    cached = {
                        "summary": str(meta.get("summary", "") or "").strip(),
                        "entities": str(meta.get("entities", "") or "").strip(),
                        "document_type": str(
                            meta.get("document_type", "") or ""
                        ).strip(),
                        "document_category": str(
                            meta.get("document_category", "") or ""
                        ).strip(),
                        "location": str(meta.get("location", "") or "").strip(),
                        "topics": str(meta.get("topics", "") or "").strip(),
                    }
                    break

        if cached is not None:
            if not cached["location"]:
                cached["location"] = infer_location_from_source(source)

            generated[source] = cached
            reused_count += 1

            print(f"  Using cached metadata: {source}")

            for meta in data["metadatas"]:
                meta["metadata_fingerprint"] = fingerprint

            continue

        # Metadata is genuinely missing/incomplete.
        text_parts = []
        current_length = 0

        for chunk_text in data["texts"]:
            chunk_text = str(chunk_text or "").strip()

            if not chunk_text:
                continue

            remaining = max_chars_per_document - current_length

            if remaining <= 0:
                break

            selected = chunk_text[:remaining]
            text_parts.append(selected)
            current_length += len(selected)

        document_text = "\n\n".join(text_parts)

        print(f"  Generating metadata for: {source}")

        try:
            if document_text:
                result = document_metadata_chain.invoke(
                    {"document_text": document_text}
                )

                summary = str(result.summary or "").strip()
                entities = _normalise_metadata_list(result.entities)
                document_type = str(result.document_type or "").strip()
                document_category = str(result.document_category or "").strip()
                location = str(result.location or "").strip()
                topics = _normalise_metadata_list(result.topics)
            else:
                summary = ""
                entities = ""
                document_type = ""
                document_category = ""
                location = ""
                topics = ""

            if not location:
                location = infer_location_from_source(source)

            generated[source] = {
                "summary": summary,
                "entities": entities,
                "document_type": document_type,
                "document_category": document_category,
                "location": location,
                "topics": topics,
            }

            generated_count += 1

        except Exception as e:
            print(f"  Warning: could not generate metadata for {source}: {e}")

            generated[source] = {
                "summary": "",
                "entities": "",
                "document_type": "",
                "document_category": "",
                "location": infer_location_from_source(source),
                "topics": "",
            }

    # Persist metadata and fingerprint to every chunk.
    for source, metadata in generated.items():
        source_ids = grouped[source]["ids"]
        fingerprint = _metadata_fingerprint(source, grouped[source]["texts"])

        updated_metadatas = []

        for meta in grouped[source]["metadatas"]:
            updated = dict(meta)

            for key, value in metadata.items():
                if value:
                    updated[key] = value

            updated["metadata_fingerprint"] = fingerprint

            updated_metadatas.append(updated)

        try:
            collection.update(ids=source_ids, metadatas=updated_metadatas)
        except Exception as e:
            print(f"  Warning: could not save metadata for {source}: {e}")

    print(
        f"  Metadata summary: {reused_count} reused, "
        f"{generated_count} generated."
    )

    if generated_count == 0:
        print("  No LLM metadata generation was required.")

    return generated


def get_document_metadata(collection) -> dict:
    """
    Read cached document metadata only.

    This function never calls the LLM.
    """
    all_items = collection.get(include=["metadatas"])

    documents = {}

    for meta in all_items.get("metadatas", []):
        if not meta:
            continue

        source = meta.get("source_file")
        if not source:
            continue

        if source not in documents:
            documents[source] = dict(meta)

    return documents
