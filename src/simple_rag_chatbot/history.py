"""Simple in-memory conversation history."""

from __future__ import annotations


def format_history(history) -> str:
    """Convert conversation history into a string suitable for LangChain prompts."""

    if not history:
        return "No previous conversation."

    parts = []

    for turn in history:
        parts.append(f"User: {turn['user']}\nAssistant: {turn['assistant']}")

    return "\n\n".join(parts)


def add_to_history(history, question, answer, max_turns) -> None:
    """Add a conversation turn and retain only the latest max_turns turns."""

    history.append({"user": question, "assistant": answer})

    if len(history) > max_turns:
        del history[:-max_turns]
