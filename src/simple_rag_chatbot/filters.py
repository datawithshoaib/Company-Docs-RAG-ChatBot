"""Turn user-provided or LLM-extracted filter hints into ChromaDB
``where`` clauses, and the deterministic (non-LLM) category detector."""

from __future__ import annotations

from .vector_store import list_documents, resolve_source_file


def resolve_filter(collection, filter_text: str):
    """
    Match a partial filename against available source files
    and return a ChromaDB where clause. Used by the /filter command.
    """

    all_sources = list_documents(collection)

    matches = [s for s in all_sources if filter_text.lower() in s.lower()]

    if not matches:
        print(f"No documents match '{filter_text}'. Use /docs to see available files.")
        return None

    if len(matches) == 1:
        print(f"Filter set: searching only '{matches[0]}'")
        return {"source_file": matches[0]}

    print(f"Filter set: searching {len(matches)} matching files: {', '.join(matches)}")
    return {"source_file": {"$in": matches}}


def infer_query_document_category(question):
    """
    Detect broad document domain from the user's current question.
    This is intentionally deterministic and conservative.
    """
    q = str(question or "").lower()

    holiday_terms = [
        "holiday", "holidays", "public holiday",
        "company holiday", "holiday calendar",
        "leave calendar", "festival holiday",
    ]
    if any(term in q for term in holiday_terms):
        return "holiday"

    if "faq" in q or "frequently asked" in q:
        return "faq"

    if any(term in q for term in ["policy", "policy document"]):
        return "policy"

    if any(term in q for term in [
        "procedure", "process document", "how do i", "how to"
    ]):
        return "procedure"

    return None


def resolve_metadata_filters(extracted_filters, available_metadata, all_sources):
    """
    Validate extracted metadata against metadata actually
    available in Chroma.

    Returns a Chroma-compatible where clause or None.
    """

    if not extracted_filters:
        return None

    valid_filters = {}

    for key, requested_value in extracted_filters.items():
        if requested_value is None:
            continue

        requested_value = str(requested_value).strip()

        if not requested_value:
            continue

        # SOURCE FILE
        if key == "source_file":
            resolved_source = resolve_source_file(requested_value, all_sources)

            if resolved_source:
                valid_filters["source_file"] = resolved_source

            continue

        # OTHER METADATA
        if key not in available_metadata:
            continue

        requested_lower = requested_value.lower()

        matches = [
            actual_value
            for actual_value in available_metadata[key]
            if requested_lower == actual_value.lower()
        ]

        if len(matches) == 1:
            valid_filters[key] = matches[0]

    if not valid_filters:
        return None

    if len(valid_filters) == 1:
        key, value = next(iter(valid_filters.items()))
        return {key: value}

    return {"$and": [{key: value} for key, value in valid_filters.items()]}


def format_available_metadata(available_metadata, max_values_per_field: int = 100) -> str:
    """
    Convert available metadata into a compact string
    for the metadata extraction LLM.
    """

    if not available_metadata:
        return "No metadata is available."

    parts = []

    for key, values in available_metadata.items():
        display_values = values[:max_values_per_field]
        values_text = ", ".join(display_values)

        if len(values) > max_values_per_field:
            values_text += ", ..."

        parts.append(f"{key}: {values_text}")

    return "\n".join(parts)
