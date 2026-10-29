from __future__ import annotations

"""
Configuration for the RAG Chatbot.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ChromaDB
    chroma_path: str = "./chroma_db"
    collection: str = "documents"
    summary_collection: str = "documents_summary_vectors"
    metadata_collection: str = "documents_metadata_vectors"

    # models
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    no_rerank: bool = False

    groq_model: str = "openai/gpt-oss-20b"
    key: str = field(default="", repr=False)

    # retrieval
    top_k: int = 5
    candidate_k: int = 20
    no_multivector: bool = False

    # conversation
    history_turns: int = 6

    debug: bool = False

    def __post_init__(self) -> None:
        if not self.key:
            self.key = os.environ.get("GROQ_API_KEY", "")

        if not self.key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to a .env file or set GROQ_API_KEY environment variable."
            )
