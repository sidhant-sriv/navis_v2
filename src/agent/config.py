"""Configuration for the todo agent."""

from typing import Optional, Dict, Any
from dataclasses import dataclass
import os


@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector database."""

    host: str = "localhost"
    port: int = 6333
    todo_collection_name: str = "todo_memories"
    user_profile_collection_name: str = "user_profiles"
    vector_size: int = 1024  # Match mxbai-embed-large dimensions
    api_key: Optional[str] = None
    url: Optional[str] = None
    
    # Legacy property for backward compatibility
    @property
    def collection_name(self) -> str:
        """Legacy collection name - defaults to todo collection."""
        return self.todo_collection_name


@dataclass
class OllamaConfig:
    """Configuration for Ollama LLM."""

    model: str = "qwen2.5:14b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 2048


@dataclass
class Mem0Config:
    """Configuration for Mem0 memory system."""

    history_db_path: str = "./mem0_history.db"
    version: str = "v1.1"
    embedding_model: str = "ollama/bge-m3:latest"


@dataclass
class AgentConfig:
    """Main configuration for the todo agent."""

    qdrant: QdrantConfig
    ollama: OllamaConfig
    mem0: Mem0Config

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create configuration from environment variables."""
        return cls(
            qdrant=QdrantConfig(
                host=os.getenv("QDRANT_HOST", "localhost"),
                port=int(os.getenv("QDRANT_PORT", "6333")),
                todo_collection_name=os.getenv("QDRANT_TODO_COLLECTION", "todo_memories"),
                user_profile_collection_name=os.getenv("QDRANT_USER_PROFILE_COLLECTION", "user_profiles"),
                api_key=os.getenv("QDRANT_API_KEY"),
                url=os.getenv("QDRANT_URL"),
            ),
            ollama=OllamaConfig(
                model=os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct-q8_0"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("OLLAMA_MAX_TOKENS", "2048")),
            ),
            mem0=Mem0Config(
                history_db_path=os.getenv("MEM0_HISTORY_PATH", "./mem0_history.db"),
                version=os.getenv("MEM0_VERSION", "v1.1"),
                embedding_model=os.getenv(
                    "MEM0_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
                ),
            ),
        )

    def get_mem0_config(self) -> Dict[str, Any]:
        """Get Mem0 configuration dictionary."""
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": self.qdrant.todo_collection_name,
                    "host": self.qdrant.host,
                    "port": self.qdrant.port,
                    "embedding_model_dims": self.qdrant.vector_size,
                },
            },
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": self.ollama.model,
                    "temperature": self.ollama.temperature,
                    "max_tokens": self.ollama.max_tokens,
                    "ollama_base_url": self.ollama.base_url,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": "bge-m3:latest",
                    "ollama_base_url": self.ollama.base_url,
                },
            },
            "history_db_path": self.mem0.history_db_path,
            "version": self.mem0.version,
        }

        # Add API key if available
        if self.qdrant.api_key:
            config["vector_store"]["config"]["api_key"] = self.qdrant.api_key

        # Use URL if available (for Qdrant Cloud)
        if self.qdrant.url:
            config["vector_store"]["config"]["url"] = self.qdrant.url
            del config["vector_store"]["config"]["host"]
            del config["vector_store"]["config"]["port"]

        return config
