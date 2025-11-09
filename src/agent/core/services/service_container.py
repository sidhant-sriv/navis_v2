"""Service container for dependency injection."""

from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig

from src.agent.config import AgentConfig

from .memory_service import MemoryService
from .response_service import ResponseService
from .tool_service import ToolService


class ServiceContainer:
    """Container for managing service dependencies."""

    def __init__(self, config: AgentConfig):
        self._config = config
        self._memory_service = None
        self._tool_service = None
        self._response_service = None

    @property
    def memory(self) -> MemoryService:
        """Lazy initialization of memory service."""
        if self._memory_service is None:
            self._memory_service = MemoryService(self._config)
        return self._memory_service

    @property
    def tools(self) -> ToolService:
        """Lazy initialization of tool service."""
        if self._tool_service is None:
            # Tool service needs the memory instance from memory service
            self._tool_service = ToolService(self.memory.memory, self._config)
        return self._tool_service

    @property
    def responses(self) -> ResponseService:
        """Lazy initialization of response service."""
        if self._response_service is None:
            self._response_service = ResponseService(self._config)
        return self._response_service

    @property
    def config(self) -> AgentConfig:
        """Get the configuration."""
        return self._config

    @classmethod
    def from_config(cls, config: RunnableConfig) -> "ServiceContainer":
        """Create service container from LangGraph config."""
        configuration = config.get("configurable", {})
        agent_config = configuration.get("agent_config")

        if not agent_config:
            agent_config = AgentConfig.from_env()

        return cls(agent_config)

    @classmethod
    def from_agent_config(cls, agent_config: AgentConfig) -> "ServiceContainer":
        """Create service container from agent config."""
        return cls(agent_config)

    def get_user_id(
        self, config: RunnableConfig, state: Optional[Dict[str, Any]] = None
    ) -> str:
        """Extract user ID from config or state."""
        configuration = config.get("configurable", {})
        user_id = configuration.get("user_id")

        if not user_id and state:
            user_id = state.get("user_id")

        return user_id or "default_user"
