"""
The RAG workflow, as a LangGraph state machine:

    START
      |
      v
   intent
      |
      +----> direct_response ----> END
      |
      v
   metadata
      |
      v
    rewrite
      |
      v
   retrieve
      |
      v
    rerank
      |
      v
   generate
      |
      v
     END

Direct-response intents (out_of_scope, document_list, document_filter)
terminate early without touching the vector store or the RAG prompt.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .context import build_context
from .filters import infer_query_document_category, resolve_metadata_filters
from .vector_store import multivector_retrieve


class RAGGraphState(TypedDict, total=False):
    question: str
    history_text: str
    document_list_text: str
    metadata_text: str

    intent: str
    intent_confidence: float

    metadata_filters: dict[str, str]
    metadata_confidence: float
    automatic_filter: Any
    retrieval_filter: Any

    query_category: str
    standalone_query: str

    query_embedding: Any

    docs: list[str]
    metas: list[dict]
    ranked: list

    answer: str
    sources: list[str]

    route: str
    error: str


def build_langgraph_rag(
    *,
    intent_chain,
    metadata_chain,
    query_rewrite_chain,
    rag_chain,
    collection,
    embedder,
    reranker,
    args,
    all_sources,
    available_metadata,
    summary_vector_collection,
    metadata_vector_collection,
    active_filter,
):
    """
    Build the end-to-end RAG workflow as a LangGraph.

    The graph is rebuilt per user turn because ``active_filter`` is
    user-controlled session state (set via the /filter command) that
    the closures below need to see fresh on every call.
    """

    class GraphState(RAGGraphState):
        pass

    workflow = StateGraph(GraphState)

    # --------------------------------------------------------
    # NODE 1: INTENT
    # --------------------------------------------------------
    def classify_intent(state: RAGGraphState):
        try:
            result = intent_chain.invoke({
                "history": state.get("history_text", ""),
                "documents": state.get("document_list_text", ""),
                "question": state["question"],
            })

            intent = result.intent
            confidence = float(result.confidence)

            print(f"[Intent: {intent} | Confidence: {confidence:.2f}]")

            return {"intent": intent, "intent_confidence": confidence}

        except Exception as e:
            return {
                "intent": "out_of_scope",
                "intent_confidence": 0.0,
                "answer": (
                    "I couldn't determine the type of question from "
                    "the available documentation."
                ),
                "route": "direct_response",
                "error": str(e),
            }

    # --------------------------------------------------------
    # ROUTER AFTER INTENT
    # --------------------------------------------------------
    def route_after_intent(state: RAGGraphState):
        intent = state.get("intent")

        if intent == "document_question":
            return "metadata"

        return "direct_response"

    # --------------------------------------------------------
    # NODE 2: DIRECT RESPONSE
    # --------------------------------------------------------
    def direct_response(state: RAGGraphState):
        intent = state.get("intent")

        if state.get("answer"):
            return {}

        if intent == "out_of_scope":
            answer = "That question is outside the scope of the available documentation."

        elif intent == "document_list":
            answer = "The available documents are:\n" + "\n".join(
                f"- {source}" for source in all_sources
            )

        elif intent == "document_filter":
            answer = (
                "Please use the /filter command to explicitly restrict "
                "the search to a document."
            )

        else:
            answer = "I couldn't determine how to process that question."

        return {"answer": answer, "route": "direct_response"}

    # --------------------------------------------------------
    # NODE 3: METADATA EXTRACTION
    # --------------------------------------------------------
    def extract_metadata(state: RAGGraphState):
        metadata_result = None

        try:
            metadata_result = metadata_chain.invoke({
                "history": state.get("history_text", ""),
                "documents": state.get("document_list_text", ""),
                "metadata": state.get("metadata_text", ""),
                # IMPORTANT: original user question
                "question": state["question"],
            })

        except Exception as e:
            print(f"Bot: Error extracting metadata: {e}\n")

        automatic_filter = None

        # Deterministic category detection is intentionally
        # independent of the LLM metadata extractor.
        query_category = infer_query_document_category(state["question"])

        category_filter = None

        if query_category and "document_category" in available_metadata:
            category_filter = {"document_category": query_category}

        if metadata_result:
            filters = metadata_result.filters or {}
            confidence = float(metadata_result.confidence)

            print(f"[Metadata extraction: {filters} | Confidence: {confidence:.2f}]")

            if confidence >= 0.70 and filters:
                automatic_filter = resolve_metadata_filters(
                    filters, available_metadata, all_sources
                )

        if category_filter:
            if automatic_filter:
                automatic_filter = {"$and": [automatic_filter, category_filter]}
            else:
                automatic_filter = category_filter

            print(f"[Document category detected: {query_category}]")

        if automatic_filter:
            print(f"[Automatic metadata filter: {automatic_filter}]")
        elif metadata_result and metadata_result.filters:
            print("[Automatic metadata filter: none — extracted metadata could not be matched]")

        return {
            "metadata_filters": metadata_result.filters if metadata_result else {},
            "metadata_confidence": (
                float(metadata_result.confidence) if metadata_result else 0.0
            ),
            "automatic_filter": automatic_filter,
            "query_category": query_category or "",
        }

    # --------------------------------------------------------
    # NODE 4: QUERY REWRITE
    # --------------------------------------------------------
    def rewrite_query(state: RAGGraphState):
        try:
            response = query_rewrite_chain.invoke({
                "history": state.get("history_text", ""),
                "question": state["question"],
            })

            standalone_query = response.content.strip()

        except Exception as e:
            return {
                "answer": "I couldn't prepare the question for document retrieval.",
                "error": str(e),
                "route": "direct_response",
            }

        print(f"[Search query: {standalone_query}]")

        retrieval_filter = None

        # Explicit /filter always wins.
        if active_filter:
            retrieval_filter = active_filter
            print("[Retrieval filter: explicit /filter]")

        elif state.get("automatic_filter"):
            retrieval_filter = state["automatic_filter"]
            print("[Retrieval filter: automatic metadata]")

        else:
            print("[Retrieval filter: none — normal semantic retrieval]")

        return {"standalone_query": standalone_query, "retrieval_filter": retrieval_filter}

    # --------------------------------------------------------
    # NODE 5: MULTI-VECTOR RETRIEVAL
    # --------------------------------------------------------
    def retrieve(state: RAGGraphState):
        if state.get("answer"):
            return {}

        query = state["standalone_query"]

        try:
            query_embedding = embedder.encode([query]).tolist()
        except Exception as e:
            return {
                "answer": "I couldn't generate the retrieval embedding for that question.",
                "error": str(e),
                "route": "direct_response",
            }

        n_results = args.candidate_k if reranker else args.top_k

        docs = []
        metas = []

        if (
            not args.no_multivector
            and summary_vector_collection is not None
            and metadata_vector_collection is not None
        ):
            try:
                (fused, content_results, summary_results, metadata_results) = (
                    multivector_retrieve(
                        collection,
                        summary_vector_collection,
                        metadata_vector_collection,
                        query_embedding,
                        n_results,
                        state.get("retrieval_filter"),
                    )
                )

                print("\n[Multi-vector candidates]")
                print(f"  Content vector: {len(content_results)}")
                print(f"  Summary vector: {len(summary_results)}")
                print(f"  Entity/topic vector: {len(metadata_results)}")
                print(f"  Fused candidates: {len(fused)}")

                docs = [item["document"] for item in fused]
                metas = [item["metadata"] for item in fused]

            except Exception as e:
                print(f"Warning: multi-vector retrieval failed: {e}")

        # Fallback to primary content vector.
        if not docs:
            query_kwargs = {"query_embeddings": query_embedding, "n_results": n_results}

            if state.get("retrieval_filter"):
                query_kwargs["where"] = state["retrieval_filter"]

            try:
                results = collection.query(**query_kwargs)
                docs = results["documents"][0]
                metas = results["metadatas"][0]

            except Exception as e:
                return {
                    "docs": [],
                    "metas": [],
                    "answer": "I couldn't retrieve information from the document index.",
                    "error": str(e),
                    "route": "direct_response",
                }

        # If the automatic filter returned no chunks,
        # perform an unfiltered semantic fallback.
        if not docs and state.get("automatic_filter") and not active_filter:
            print(
                "[Automatic metadata filter returned no chunks. "
                "Falling back to normal semantic retrieval.]"
            )

            try:
                if (
                    not args.no_multivector
                    and summary_vector_collection is not None
                    and metadata_vector_collection is not None
                ):
                    (fused, _, _, _) = multivector_retrieve(
                        collection,
                        summary_vector_collection,
                        metadata_vector_collection,
                        query_embedding,
                        n_results,
                        None,
                    )

                    docs = [item["document"] for item in fused]
                    metas = [item["metadata"] for item in fused]

                else:
                    results = collection.query(
                        query_embeddings=query_embedding, n_results=n_results
                    )
                    docs = results["documents"][0]
                    metadatas = results["metadatas"][0]

            except Exception as e:
                return {
                    "docs": [],
                    "metas": [],
                    "answer": "I couldn't retrieve relevant documentation.",
                    "error": str(e),
                    "route": "direct_response",
                }

        print("\n[Chroma candidates]")
        for meta in metas:
            print(f"  {meta.get('source_file', 'unknown')}")

        return {"query_embedding": query_embedding, "docs": docs, "metas": metas}

    # --------------------------------------------------------
    # NODE 6: RERANK
    # --------------------------------------------------------
    def rerank(state: RAGGraphState):
        if state.get("answer"):
            return {}

        docs = state.get("docs", [])
        metas = state.get("metas", [])

        if not docs:
            return {
                "answer": (
                    "I couldn't find information in the available "
                    "documentation to answer that question."
                ),
                "route": "direct_response",
            }

        if not reranker:
            return {"docs": docs[: args.top_k], "metas": metas[: args.top_k]}

        pairs = [[state["standalone_query"], doc] for doc in docs]

        try:
            scores = reranker.predict(pairs)
        except Exception as e:
            return {
                "answer": "I found relevant documents, but the relevance ranking step failed.",
                "error": str(e),
                "route": "direct_response",
            }

        ranked = sorted(zip(scores, docs, metas), key=lambda x: x[0], reverse=True)

        print("\n[Retrieval scores]")
        for score, _, meta in ranked:
            print(f"  {score:.4f} | {meta.get('source_file', 'unknown')}")

        relevant = ranked[: args.top_k]

        return {
            "ranked": ranked,
            "docs": [doc for _, doc, _ in relevant],
            "metas": [meta for _, _, meta in relevant],
        }

    # --------------------------------------------------------
    # NODE 7: GENERATE
    # --------------------------------------------------------
    def generate_answer(state: RAGGraphState):
        if state.get("answer"):
            return {}

        docs = state.get("docs", [])
        metas = state.get("metas", [])

        if not docs:
            return {
                "answer": (
                    "I couldn't find information in the available "
                    "documentation to answer that question."
                )
            }

        context = build_context(docs, metas)

        try:
            response = rag_chain.invoke({
                "history": state.get("history_text", ""),
                "context": context,
                "question": state["question"],
            })

            answer = response.content.strip()

        except Exception as e:
            return {
                "answer": (
                    "I found relevant documentation, but could not "
                    "generate the final answer."
                ),
                "error": str(e),
            }

        sources = []
        for meta in metas:
            source = meta.get("source_file", "unknown")
            if source not in sources:
                sources.append(source)

        return {"answer": answer, "sources": sources}

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------
    workflow.add_node("intent", classify_intent)
    workflow.add_node("direct_response", direct_response)
    workflow.add_node("metadata", extract_metadata)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rerank", rerank)
    workflow.add_node("generate", generate_answer)

    workflow.add_edge(START, "intent")

    workflow.add_conditional_edges(
        "intent",
        route_after_intent,
        {"metadata": "metadata", "direct_response": "direct_response"},
    )

    workflow.add_edge("metadata", "rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", END)
    workflow.add_edge("direct_response", END)

    return workflow.compile()
