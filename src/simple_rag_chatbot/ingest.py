"""
A deliberately simple ingestion pipeline for the workshop: read PDFs
and text files from a folder, chunk them, embed the chunks, and load
them into a Chroma collection with the metadata fields the rest of
the codebase expects (``source_file``, ``page_numbers``).

This is *not* the original ``batch_extract_to_chroma.py`` (which
wasn't available) — it's a small, readable replacement good enough for
interns to run end-to-end and then improve as an exercise (better
chunking, OCR, table extraction, etc.).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import chromadb


def _read_pdf(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, page_text), ...] for a PDF file."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def _read_text(path: Path) -> list[tuple[int, str]]:
    """Return [(1, full_text)] for a plain text / markdown file."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(1, text)] if text.strip() else []


def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Simple sliding-window character chunker.

    Good enough for a workshop; a production system would chunk on
    sentence/paragraph boundaries and respect token limits instead of
    character counts.
    """
    text = " ".join(text.split())  # normalize whitespace

    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap

    return chunks


def load_documents_into_chroma(
    docs_dir: str | Path,
    chroma_path: str,
    collection_name: str,
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 1000,
    overlap: int = 150,
    embedder=None,
) -> "chromadb.Collection":
    """
    Read every .pdf/.txt/.md file in ``docs_dir``, chunk it, embed the
    chunks, and upsert them into a persistent Chroma collection at
    ``chroma_path``. Safe to re-run: chunk ids are content hashes, so
    re-running on unchanged files is a no-op (upsert).
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents folder not found: {docs_dir}")

    if embedder is None:
        embedder = SentenceTransformer(embed_model_name)

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(collection_name)

    files = sorted(
        [
            p
            for p in docs_dir.iterdir()
            if p.suffix.lower() in (".pdf", ".txt", ".md")
        ]
    )

    if not files:
        raise FileNotFoundError(
            f"No .pdf/.txt/.md files found in {docs_dir}. Add some documents "
            "and re-run."
        )

    total_chunks = 0

    for path in files:
        print(f"Reading {path.name}...")

        if path.suffix.lower() == ".pdf":
            pages = _read_pdf(path)
        else:
            pages = _read_text(path)

        ids, texts, metadatas = [], [], []

        for page_number, page_text in pages:
            for chunk in _chunk_text(page_text, chunk_size, overlap):
                chunk_id = hashlib.sha256(
                    f"{path.name}:{page_number}:{chunk}".encode("utf-8")
                ).hexdigest()

                ids.append(chunk_id)
                texts.append(chunk)
                metadatas.append(
                    {"source_file": path.name, "page_numbers": str(page_number)}
                )

        if not ids:
            print(f"  Warning: no extractable text in {path.name}, skipping.")
            continue

        embeddings = embedder.encode(texts).tolist()

        collection.upsert(
            ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
        )

        print(f"  Added {len(ids)} chunks from {path.name}")
        total_chunks += len(ids)

    print(f"\nDone. {total_chunks} chunks in collection '{collection_name}'.")
    return collection
