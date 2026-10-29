from __future__ import annotations

"""Terminal chat loop: /docs, /filter, /history commands + the graph"""

from typing import TYPE_CHECKING

from .config import Config
from .filters import format_available_metadata, resolve_filter
from .graph import build_langgraph_rag
from .history import add_to_history, format_history
from .metadata_enrichment import get_document_metadata
from .session import build_session

if TYPE_CHECKING:
    from .session import Session


def print_commands() -> None:
    print(
        "\nCommands:"
        "\n/docs"
        "\n/filter <text>"
        "\n/filter clear"
        "\n/history"
        "\n/history clear"
        "\nexit / quit\n"
    )


def chat_loop(session: Session) -> None:
    print_commands()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue

        if question.lower() in ["exit", "quit"]:
            print("\nBye.")
            break

        if question == "/docs":
            document_metadata = get_document_metadata(session.collection)
            print("\nAvailable documents: \n")
            for i, source in enumerate(session.all_sources, start=1):
                info = document_metadata.get(source, {})
                print(f"{i}. {source}")
                for field, label in [
                    ("summary", "Summary"),
                    ("entities", "Entities"),
                    ("document_type", "Type"),
                    ("document_category", "Category"),
                    ("location", "Location"),
                    ("topics", "Topics"),
                ]:
                    value = info.get(field, "")
                    if value:
                        print(f"  {label}: {value}")
                print()
            continue

        if question == "/history":
            if not session.conversation_history:
                print("\nNo conversation history yet.\n")
            else:
                print("\nConversation History:\n")

                for i, turn in enumerate(session.conversation_history, start=1):
                    print(f"\n[{i}] User: \n{turn['user']}")
                    print(f"\nAssistant: \n{turn['assistant']}")

                print()
            continue

        if question == "/history clear":
            session.conversation_history = []
            print("\nConversation history cleared.\n")
            continue

        if question.lower().startswith("/filter"):
            arg = question[len("/filter"):].strip()

            if not arg or arg.lower() == "clear":
                session.active_filter = None
                print("Filter cleared - searching all documents again.\n")
            else:
                session.active_filter = resolve_filter(session.collection, arg)
                print()

            continue

        history_text = format_history(session.conversation_history)
        document_list_text = "\n".join(f"- {source}" for source in session.all_sources)
        metadata_text = format_available_metadata(session.available_metadata)

        try:
            rag_graph = build_langgraph_rag(
                intent_chain=session.chains["intent_chain"],
                metadata_chain=session.chains["metadata_chain"],
                query_rewrite_chain=session.chains["query_rewrite_chain"],
                rag_chain=session.chains["rag_chain"],
                collection=session.collection,
                embedder=session.embedder,
                reranker=session.reranker,
                args=session.config,
                all_sources=session.all_sources,
                available_metadata=session.available_metadata,
                summary_vector_collection=session.summary_vector_collection,
                metadata_vector_collection=session.metadata_vector_collection,
                active_filter=session.active_filter,
            )

            graph_result = rag_graph.invoke({
                "question": question,
                "history_text": history_text,
                "document_list_text": document_list_text,
                "metadata_text": metadata_text,
            })

        except Exception as e:
            print(f"Bot: LangGraph workflow failed: {e}\n")
            continue

        answer = graph_result.get("answer", "I couldn't generate an answer.")
        sources = graph_result.get("sources", [])

        print(f"\nBot: {answer}")
        if sources:
            print(f"\n[Sources: {', '.join(sources)}]")
        print()

        add_to_history(
            session.conversation_history, question, answer, session.config.history_turns
        )


def main() -> None:
    config = Config()
    session = build_session(config)
    chat_loop(session)


if __name__ == "__main__":
    main()
