"""Backward-compatibility alias for simple_rag_chatbot.vector_store."""

from __future__ import annotations

from .vector_store import (
    build_multivector_indexes,
    get_available_metadata,
    list_documents,
    multivector_retrieve,
    resolve_source_file,
)

__all__ = [
    "build_multivector_indexes",
    "get_available_metadata",
    "list_documents",
    "multivector_retrieve",
    "resolve_source_file",
]

