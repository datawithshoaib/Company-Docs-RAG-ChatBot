"""Structured-output schemas used by the LangChain chains."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QueryIntent(BaseModel):
    """Structured output from the intent classifier."""

    intent: Literal[
        "document_question",
        "document_list",
        "document_filter",
        "out_of_scope",
    ] = Field(description="The intent of the user's question.")

    confidence: float = Field(description="Confidence between 0 and 1.")


class MetadataExtraction(BaseModel):
    """Structured output from metadata extraction."""

    filters: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Metadata key-value filters extracted from the "
            "user's ORIGINAL question."
        ),
    )

    confidence: float = Field(description="Confidence between 0 and 1.")


class DocumentMetadata(BaseModel):
    """Structured metadata generated for each source document."""

    summary: str = Field(
        description=(
            "A concise one-line summary explaining what the "
            "document is about. Keep it between 10 and 30 words."
        )
    )

    entities: list[str] = Field(
        default_factory=list,
        description=(
            "Important entities explicitly supported by the document, "
            "such as locations, organizations, departments, policies, "
            "systems, dates, business areas, or document subjects."
        ),
    )

    document_type: str = Field(
        default="",
        description=(
            "The type of document, for example Holiday Calendar, "
            "Policy, Procedure, Technical Document, Report, or Manual."
        ),
    )

    document_category: str = Field(
        default="other",
        description=(
            "A normalized document category. Use one of: holiday, "
            "policy, procedure, faq, technical, report, manual, "
            "training, other."
        ),
    )

    location: str = Field(
        default="",
        description=(
            "Primary location explicitly associated with the document, "
            "if one exists. Otherwise return an empty string."
        ),
    )

    topics: list[str] = Field(
        default_factory=list,
        description="The main topics covered by the document.",
    )
