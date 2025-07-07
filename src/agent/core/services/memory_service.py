"""Memory service for centralized memory operations."""

from typing import List, Optional
from mem0 import Memory

from src.agent.config import AgentConfig
from src.agent.models import UserInfo, UserContext
from src.agent.memory_adapters import create_memory_adapter


class MemoryService:
    """Centralized service for all memory operations."""
    
    def __init__(self, config: AgentConfig):
        self._config = config
        self._memory = None
        self._adapter = None
    
    @property
    def memory(self) -> Memory:
        """Lazy initialization of Mem0 memory."""
        if self._memory is None:
            mem0_config = self._config.get_mem0_config()
            self._memory = Memory.from_config(mem0_config)
        return self._memory
    
    @property
    def adapter(self):
        """Lazy initialization of memory adapter."""
        if self._adapter is None:
            self._adapter = create_memory_adapter(self._config)
        return self._adapter
    
    def get_user_context(self, user_id: str) -> UserContext:
        """Get comprehensive user context for enhanced interactions."""
        try:
            return self.adapter.get_user_context(user_id)
        except Exception as e:
            print(f"DEBUG: Failed to retrieve user context: {e}")
            # Return empty context instead of raising
            return UserContext(user_id=user_id)
    
    def get_user_context_summary(self, user_id: str) -> str:
        """Get user context as a summary string."""
        try:
            user_context = self.get_user_context(user_id)
            summary = user_context.get_context_summary()
            return summary if summary != "No user context available" else ""
        except Exception as e:
            print(f"DEBUG: Failed to get user context summary: {e}")
            return ""
    
    def search_user_info(self, user_id: str, query: str, limit: int = 10) -> List[UserInfo]:
        """Search for relevant user information."""
        try:
            return self.adapter.search_user_info(user_id, query, limit=limit)
        except Exception as e:
            print(f"DEBUG: Failed to search user info: {e}")
            return []
    
    def extract_and_store_user_info(self, conversation_text: str, user_id: str) -> List[UserInfo]:
        """Extract user information from conversation and store it."""
        try:
            return self.adapter.extract_and_store_user_info(conversation_text, user_id)
        except Exception as e:
            print(f"DEBUG: Failed to extract and store user info: {e}")
            return []
    
    def should_extract_info_from_message(self, message: str, state_extracted: bool = False) -> bool:
        """Determine if we should extract user info from this message."""
        return bool(
            message and 
            len(message.strip()) > 10 and 
            not state_extracted  # Don't extract if we already did for this turn
        ) 