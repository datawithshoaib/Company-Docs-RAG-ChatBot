from __future__ import annotations

from langchain_groq import ChatGroq

from .models import DocumentMetadata, MetadataExtraction, QueryIntent
from .prompts import (
    DOCUMENT_METADATA_PROMPT,
    INTENT_PROMPT,
    METADATA_EXTRACTION_PROMPT,
    QUERY_REWRITE_PROMPT,
    RAG_PROMPT,
)


def build_llm(config) -> ChatGroq:
    return ChatGroq(
        model=config.groq_model,
        api_key=config.key,
        temperature=0,
        max_tokens=1200,
    )


def build_chains(llm: ChatGroq) -> dict:
    """Returns a dictionary of five chains used across the graph:
    intent classification, metadata extraction, per-document metadata generation,
    query rewriting, and final answer generation."""

    return {
        "intent_chain": INTENT_PROMPT | llm.with_structured_output(QueryIntent),
        "metadata_chain": (
            METADATA_EXTRACTION_PROMPT | llm.with_structured_output(MetadataExtraction)
        ),
        "document_metadata_chain": (
            DOCUMENT_METADATA_PROMPT | llm.with_structured_output(DocumentMetadata)
        ),
        "query_rewrite_chain": QUERY_REWRITE_PROMPT | llm,
        "rag_chain": RAG_PROMPT | llm,
    }
