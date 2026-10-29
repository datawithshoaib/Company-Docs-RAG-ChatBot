"""Turn retrieved chunks into the context block handed to the LLM."""

from __future__ import annotations


def build_context(docs, metas) -> str:
    """Convert retrieved documents and metadata into the context
    supplied to the LLM."""

    parts = []

    for doc, meta in zip(docs, metas):
        source = meta.get("source_file", "unknown")
        pages = meta.get("page_numbers", "")
        heading = meta.get("headings", "")

        label = (
            f"[Source: {source}"
            + (f", pages {pages}" if pages else "")
            + (f", section: {heading}" if heading else "")
            + "]"
        )

        parts.append(f"{label}\n{doc}")

    return "\n\n---\n\n".join(parts)
