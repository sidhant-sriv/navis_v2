"""Tool service for centralized tool operations."""

from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool
from mem0 import Memory

from src.agent.config import AgentConfig
from src.agent.tools import create_conversation_tools, create_todo_tools


class ToolService:
    """Centralized service for tool creation and management."""

    def __init__(self, memory: Memory, config: AgentConfig):
        self._memory = memory
        self._config = config
        self._todo_tools = None
        self._conversation_tools = None
        self._all_tools = None

    @property
    def todo_tools(self) -> List[BaseTool]:
        """Lazy initialization of todo tools."""
        if self._todo_tools is None:
            self._todo_tools = create_todo_tools(self._memory, self._config)
        return self._todo_tools

    @property
    def conversation_tools(self) -> List[BaseTool]:
        """Lazy initialization of conversation tools."""
        if self._conversation_tools is None:
            self._conversation_tools = create_conversation_tools(
                self._memory, self._config
            )
        return self._conversation_tools

    @property
    def all_tools(self) -> List[BaseTool]:
        """Get all available tools."""
        if self._all_tools is None:
            self._all_tools = self.todo_tools + self.conversation_tools
        return self._all_tools

    def get_tools_for_context(self, context: str) -> List[BaseTool]:
        """Get tools appropriate for the given context."""
        if context == "todo":
            return self.todo_tools
        elif context == "conversation":
            return self.conversation_tools
        else:
            return self.all_tools

    def find_tool_by_name(self, tool_name: str) -> Optional[BaseTool]:
        """Find a tool by its name."""
        for tool in self.all_tools:
            if tool.name == tool_name:
                return tool
        return None

    def get_tool_names(self) -> List[str]:
        """Get list of all available tool names."""
        return [tool.name for tool in self.all_tools]
