"""State definition for the todo agent following LangGraph best practices."""

from typing import List, Dict, Any, Optional
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

    # Todo processing results - simplified from complex extraction
    todo_results: Optional[Dict[str, Any]]

    # Memory integration - for context and retrieval
    memory_context: Optional[str]

    # Processing status - simpler than multiple flags
    processing_complete: bool
