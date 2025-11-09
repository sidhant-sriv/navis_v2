"""Adapters to bridge the new memory system with existing todo functionality."""

from typing import Any, Dict, List, Optional

from src.agent.config import AgentConfig
from src.agent.memory import MemoryConfig, MemoryItem, MemoryManager
from src.agent.models import UserContext, UserInfo, UserInfoTag, UserInfoType


class TodoMemoryAdapter:
    """Adapter that bridges the new memory system with todo functionality."""

    def __init__(self, agent_config: AgentConfig):
        # Convert AgentConfig to MemoryConfig
        memory_config = MemoryConfig(
            qdrant_host=agent_config.qdrant.host,
            qdrant_port=agent_config.qdrant.port,
            qdrant_api_key=agent_config.qdrant.api_key,
            qdrant_url=agent_config.qdrant.url,
            embedding_base_url=agent_config.ollama.base_url,
            embedding_model="bge-m3:latest",
            vector_size=agent_config.qdrant.vector_size,
            llm_model=agent_config.ollama.model,
            llm_base_url=agent_config.ollama.base_url,
            llm_temperature=agent_config.ollama.temperature,
            llm_max_tokens=agent_config.ollama.max_tokens,
        )

        self._memory = MemoryManager(memory_config)
        self._todo_collection = agent_config.qdrant.todo_collection_name
        self._user_profile_collection = agent_config.qdrant.user_profile_collection_name

        # Register collections
        self._memory.register_collection(
            self._todo_collection, "Todo items and task management"
        )
        self._memory.register_collection(
            self._user_profile_collection, "User personal information and preferences"
        )

    @property
    def memory_manager(self) -> MemoryManager:
        """Access the underlying memory manager."""
        return self._memory

    def store_user_info(self, user_info: UserInfo) -> bool:
        """Store user info using the new memory system."""
        memory_item = MemoryItem(
            id=user_info.id,
            user_id=user_info.user_id,
            content=user_info.content,
            item_type=user_info.info_type.value,  # Convert enum to string
            metadata={
                "relevance_score": user_info.relevance_score,
            },
            relevance_score=user_info.relevance_score,
            tags=user_info.tags,
            created_at=user_info.created_at,
            last_used=user_info.last_used,
        )

        return self._memory.store(memory_item, self._user_profile_collection)

    def search_user_info(
        self, user_id: str, query: str, limit: int = 10
    ) -> List[UserInfo]:
        """Search user info using the new memory system."""
        memory_items = self._memory.search(
            user_id, query, self._user_profile_collection, limit=limit
        )

        return [self._memory_item_to_user_info(item) for item in memory_items]

    def get_user_context(self, user_id: str, limit: int = 20) -> UserContext:
        """Get user context using the new memory system."""
        memory_items = self._memory.get_all(
            user_id, self._user_profile_collection, limit=limit
        )

        user_context = UserContext(user_id=user_id)
        for item in memory_items:
            user_info = self._memory_item_to_user_info(item)
            user_context.add_info(user_info)

        return user_context

    def extract_and_store_user_info(
        self, conversation_text: str, user_id: str
    ) -> List[UserInfo]:
        """Extract user info from conversation using the new memory system."""
        extraction_schema = {
            "types": UserInfoType.get_all_types(),
            "tags": UserInfoTag.get_all_tags(),
        }

        memory_items = self._memory.extract_and_store(
            text=conversation_text,
            user_id=user_id,
            collection_name=self._user_profile_collection,
            extraction_schema=extraction_schema,
            context="Todo management conversation",
            dedup_limit=5,
        )

        return [self._memory_item_to_user_info(item) for item in memory_items]

    def _memory_item_to_user_info(self, memory_item: MemoryItem) -> UserInfo:
        """Convert MemoryItem to UserInfo."""
        # Parse info_type from string to enum
        try:
            info_type = UserInfoType(memory_item.item_type)
        except ValueError:
            info_type = UserInfoType.PERSONAL

        return UserInfo(
            id=memory_item.id,
            user_id=memory_item.user_id,
            info_type=info_type,
            content=memory_item.content,
            relevance_score=memory_item.relevance_score,
            tags=memory_item.tags,
            created_at=memory_item.created_at,
            last_used=memory_item.last_used,
        )


def create_memory_adapter(agent_config: AgentConfig) -> TodoMemoryAdapter:
    """Factory function to create memory adapter."""
    return TodoMemoryAdapter(agent_config)
