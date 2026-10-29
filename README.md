# Company-Docs-RAG-ChatBot

A modular, production-ready Retrieval-Augmented Generation (RAG) system for querying company documentation. Built with **LangGraph**, **ChromaDB**, **Sentence Transformers**, **Cross-Encoder reranking**, and **Groq LLMs**.

---

## Architecture Overview

The system processes queries through an end-to-end **LangGraph** state machine:

```text
              START
                │
                ▼
             Intent
           Classification
            │          │
 (document_question)  (direct response: out_of_scope, document_list, filter)
            │          │
            ▼          ▼
         Metadata     END
        Extraction
            │
            ▼
      Query Rewrite (conservative reference resolution)
            │
            ▼
      Multi-Vector Retrieval (Content + Summary + Metadata spaces)
            │
            ▼
      Reciprocal Rank Fusion (RRF)
            │
            ▼
      Cross-Encoder Reranking
            │
            ▼
      Answer Generation (strictly grounded in context)
            │
            ▼
           END
```

### Key Features

- **Document-Level Metadata Enrichment**: Automatically summarizes documents, extracts entities, topics, document types, and normalized categories (`holiday`, `policy`, `procedure`, `faq`, etc.), cached via content hashing.
- **Multi-Vector Indexing**: Queries three vector spaces simultaneously:
  1. Primary chunk content
  2. Document summaries
  3. Structured metadata (entities, topics, location, categories)
- **Reciprocal Rank Fusion (RRF)**: Fuses candidate lists across vector spaces without manual distance calibration.
- **Cross-Encoder Reranking**: Scores and reorders fused candidates before presenting them to the LLM.
- **LangGraph Orchestrator**: Explicit, observable state transitions with fallback handling.
- **Interactive Terminal CLI**: Built-in interactive chat loop with `/docs`, `/filter`, and `/history` commands.

---

## Directory Structure

```text
Company-Docs-RAG-ChatBot/
├── data/                               # Store your PDF, TXT, or MD documents here
├── notebooks/
│   ├── RAG_ChatBot_Full_Pipeline.ipynb # End-to-end interactive workshop pipeline
│   └── RAG_ChatBot_Step_by_Step.ipynb  # Granular step-by-step breakdown & experiments
├── src/
│   └── simple_rag_chatbot/
│       ├── __init__.py                 # Package exports (Config, Session, build_session, main)
│       ├── chains.py                   # LangChain LLM chains & prompt wiring
│       ├── cli.py                      # Interactive terminal chat loop & CLI entrypoint
│       ├── config.py                   # Dataclass configuration loader (.env & defaults)
│       ├── context.py                  # Context assembly for LLM prompts
│       ├── filters.py                  # ChromaDB filter resolution & category detection
│       ├── graph.py                    # LangGraph workflow definition & compiled graph
│       ├── history.py                  # In-memory conversation history management
│       ├── ingest.py                   # PDF & text ingestion pipeline
│       ├── metadata_enrichment.py      # LLM-based document metadata extraction & caching
│       ├── models.py                   # Pydantic structured output models
│       ├── prompts.py                  # Standardized prompt templates
│       ├── session.py                  # Chatbot session builder (models, DB, chains)
│       ├── vector_store.py             # ChromaDB multi-vector retrieval & RRF logic
│       └── vectorstore.py              # Backward-compatibility alias
├── .env.example                        # Template for environment variables
├── .gitignore                          # Git ignore rules (secrets, venv, Chroma DB)
├── pyproject.toml                      # Project metadata and dependencies
└── README.md                           # Documentation
```

---

## Quickstart

### 1. Prerequisites

- Python `>= 3.13`
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip`
- A [Groq API Key](https://console.groq.com/)

### 2. Installation

Clone the repository and install dependencies with `uv`:

```bash
uv sync
```

Alternatively, using standard virtual environments:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -e .
```

### 3. Environment Configuration

Copy `.env.example` to `.env` and set your Groq API key:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_actual_groq_api_key
```

### 4. Ingest Documents

Place your company documentation (`.pdf`, `.txt`, `.md`) into the `data/` folder, then run ingestion via Python or the provided notebooks:

```python
from simple_rag_chatbot.ingest import load_documents_into_chroma

load_documents_into_chroma(
    docs_dir="data",
    chroma_path="./chroma_db",
    collection_name="documents",
)
```

### 5. Launch the CLI Chatbot

Start the interactive chatbot from the terminal:

```bash
uv run simple-rag-chatbot
```

Or directly via python:

```bash
python -m simple_rag_chatbot.cli
```

---

## Interactive CLI Commands

Within the terminal chat loop:

| Command | Description |
|---|---|
| `/docs` | List all indexed documents along with cached metadata (summary, entities, category, location) |
| `/filter <query>` | Restrict search and retrieval strictly to matching documents |
| `/filter clear` | Clear active filter to search across all documents again |
| `/history` | Display current conversation history turns |
| `/history clear` | Clear conversation history |
| `exit` or `quit` | Exit the chatbot session |

---

## Notebooks

- **`notebooks/RAG_ChatBot_Full_Pipeline.ipynb`**: Walk through document ingestion, building the session, testing each pipeline node independently, and running the full LangGraph workflow.
- **`notebooks/RAG_ChatBot_Step_by_Step.ipynb`**: Detailed breakdown of document conversion (`docling`), multi-vector index generation, intent classification, metadata extraction, Reciprocal Rank Fusion, and Cross-Encoder reranking.

---

## Security & Privacy

- **No Secrets in Code**: No API keys, credentials, or confidential secrets are committed. All secrets are loaded securely from environment variables / `.env`.
- **Git Protection**: `.gitignore` is configured to prevent accidental commits of `.env` files, local ChromaDB instances (`chroma_db/`), checkpoints, and temporary data.
- **Self-Contained & Local**: Vector databases and document embeddings are stored locally in your chosen directory.

