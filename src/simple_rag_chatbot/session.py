"""
Everything needed to go from a ``Config`` to a ready-to-query chatbot
session, in one place. Both the CLI (``cli.py``) and the workshop
notebook call ``build_session`` so the two never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

from .chains import build_chains, build_llm
from .metadata_enrichment import enrich_collection_with_document_metadata
from .vector_store import build_multivector_indexes, get_available_metadata, list_documents


@dataclass
class Session:
    config: Any
    client: chromadb.ClientAPI
    collection: Any
    embedder: SentenceTransformer
    reranker: CrossEncoder | None
    llm: Any
    chains: dict
    all_sources: list[str] = field(default_factory=list)
    available_metadata: dict = field(default_factory=dict)
    summary_vector_collection: Any = None
    metadata_vector_collection: Any = None
    conversation_history: list = field(default_factory=list)
    active_filter: Any = None


def build_session(config, run_metadata_enrichment: bool = True) -> Session:
    """Load the embedding/reranking models, connect to Chroma, build
    the LangChain chains, and (optionally) enrich the collection with
    document-level metadata + multi-vector indexes."""

    print("Loading embedding model...")
    embedder = SentenceTransformer(config.embed_model)

    reranker = None
    if not config.no_rerank:
        print(f"Loading re-ranking model ({config.rerank_model})...")
        reranker = CrossEncoder(config.rerank_model)

    print(f"Connecting to ChromaDB at {config.chroma_path} (collection: {config.collection})...")
    client = chromadb.PersistentClient(path=config.chroma_path)

    try:
        collection = client.get_collection(config.collection)
    except Exception:
        raise SystemExit(
            f"Collection '{config.collection}' not found at {config.chroma_path}. "
            "Run the ingestion step first (see notebooks/RAG_ChatBot_Full_Pipeline.ipynb "
            "or simple_rag_chatbot.ingest)."
        )

    count = collection.count()
    if count == 0:
        raise SystemExit(f"Collection '{config.collection}' is empty.")

    llm = build_llm(config)
    chains = build_chains(llm)

    session = Session(
        config=config,
        client=client,
        collection=collection,
        embedder=embedder,
        reranker=reranker,
        llm=llm,
        chains=chains,
    )

    if run_metadata_enrichment:
        enrich_collection_with_document_metadata(
            collection, chains["document_metadata_chain"]
        )

    session.all_sources = list_documents(collection)
    session.available_metadata = get_available_metadata(collection)

    print(
        f"Ready. {count} chunks across {len(session.all_sources)} document(s). "
        f"(model: {config.groq_model}, top-k: {config.top_k}, "
        f"candidate-k: {config.candidate_k}, history: {config.history_turns} turns, "
        f"rerank: {'off' if config.no_rerank else 'on'})"
    )

    if not config.no_multivector:
        try:
            (
                session.summary_vector_collection,
                session.metadata_vector_collection,
            ) = build_multivector_indexes(
                client,
                collection,
                embedder,
                config.summary_collection,
                config.metadata_collection,
            )
        except Exception as e:
            print(f"Warning: multi-vector indexes could not be initialized: {e}")
            print("Continuing with the original content-vector collection only.")

    return session
