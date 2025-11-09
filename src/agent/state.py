"""State definition for the todo agent following LangGraph best practices."""

from typing import Any, Dict

from langgraph.graph import MessagesState


class TodoAgentState(MessagesState):
    """State for the todo management agent following LangGraph best practices.

    Based on LangGraph patterns from:
    https://langchain-ai.github.io/langgraph/tutorials/get-started/5-customize-state/

    This state extends MessagesState to include todo-specific fields while
    maintaining compatibility with LangGraph's message handling patterns.
    """

    # User identification - required for memory operations
    user_id: str

    # Intent routing - for determining conversation vs todo operations
    detected_intent: str | None  # "todo" or "conversation"
    # Todo processing results - simplified from complex extraction
    todo_results: Dict[str, Any] | None

    # Memory integration - for context and retrieval
    memory_context: str | None

    # User context - for enhanced todo creation
    user_context: str | None  # Summary of user information for context-aware todos
    # Processing status - simpler than multiple flags
    processing_complete: bool

    # User info extraction tracking - to prevent duplicates
    user_info_extracted: bool | None
