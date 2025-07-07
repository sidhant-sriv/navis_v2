"""Reusable memory system for storing and retrieving user information and context.

This module provides a flexible, configurable memory system that can be used
across different applications and agents.
"""

from typing import List, Optional, Dict, Any, Protocol
from dataclasses import dataclass, field
from datetime import datetime
import json
import re
import logging
import uuid
import requests

from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    UpdateStatus,
    Condition,
    VectorParams,
    Distance,
)
from langchain.chat_models import init_chat_model

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory system backends."""
    
    # Vector database config
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    qdrant_url: Optional[str] = None
    
    # Embedding config
    embedding_base_url: str = "http://localhost:11434"
    embedding_model: str = "bge-m3:latest"
    vector_size: int = 1024
    
    # LLM config for extraction
    llm_model: str = "llama3.1:8b-instruct-q8_0"
    llm_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048


@dataclass
class MemoryItem:
    """Generic memory item that can represent any type of stored information."""
    
    id: str
    user_id: str
    content: str
    item_type: str
    metadata: Dict[str, Any]
    relevance_score: float = 1.0
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    last_used: Optional[str] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class MemoryBackend(Protocol):
    """Protocol for memory storage backends."""
    
    def ensure_collection(self, collection_name: str) -> None:
        """Ensure collection exists."""
        ...
    
    def store_item(self, item: MemoryItem, collection_name: str) -> bool:
        """Store a memory item."""
        ...
    
    def search_items(
        self, 
        user_id: str, 
        query: str, 
        collection_name: str,
        item_type: Optional[str] = None,
        limit: int = 10
    ) -> List[MemoryItem]:
        """Search memory items by text query."""
        ...
    
    def get_items(
        self,
        user_id: str,
        collection_name: str,
        item_type: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[MemoryItem]:
        """Retrieve memory items with optional filtering."""
        ...
    
    def update_item(
        self,
        item_id: str,
        user_id: str,
        collection_name: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update a memory item."""
        ...
    
    def delete_item(self, item_id: str, user_id: str, collection_name: str) -> bool:
        """Delete a memory item."""
        ...


class QdrantMemoryBackend:
    """Qdrant-based memory storage backend."""
    
    def __init__(self, config: MemoryConfig):
        self._config = config
        
        # Initialize Qdrant client
        if config.qdrant_url:
            self._client = QdrantClient(
                url=config.qdrant_url,
                api_key=config.qdrant_api_key
            )
        else:
            self._client = QdrantClient(
                host=config.qdrant_host,
                port=config.qdrant_port
            )
        
        self._embedding_url = f"{config.embedding_base_url}/api/embeddings"
        self._embedding_model = config.embedding_model
        self._vector_size = config.vector_size
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        try:
            response = requests.post(
                self._embedding_url,
                json={"model": self._embedding_model, "prompt": text}
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            logger.warning(f"Failed to get embedding: {type(e).__name__}")
            return [0.0] * self._vector_size
    
    def ensure_collection(self, collection_name: str) -> None:
        """Ensure collection exists."""
        try:
            self._client.get_collection(collection_name)
            logger.debug(f"Collection '{collection_name}' already exists")
        except Exception:
            vector_config = VectorParams(
                size=self._vector_size,
                distance=Distance.COSINE
            )
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=vector_config
            )
            logger.info(f"Created collection '{collection_name}'")
    
    def store_item(self, item: MemoryItem, collection_name: str) -> bool:
        """Store a memory item in Qdrant."""
        try:
            self.ensure_collection(collection_name)
            
            # Create searchable content
            content = f"{item.item_type}: {item.content}"
            if item.tags:
                content += f" - Tags: {', '.join(item.tags)}"
            
            embedding = self._get_embedding(content)
            point_uuid = str(uuid.uuid4())
            
            payload = {
                "memory_item_id": item.id,
                "user_id": item.user_id,
                "content": item.content,
                "item_type": item.item_type,
                "metadata": item.metadata,
                "relevance_score": item.relevance_score,
                "tags": item.tags,
                "created_at": item.created_at,
                "last_used": item.last_used,
                "searchable_content": content,
            }
            
            point = PointStruct(
                id=point_uuid,
                vector=embedding,
                payload=payload,
            )
            
            result = self._client.upsert(
                collection_name=collection_name, points=[point]
            )
            
            return result.status == UpdateStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Failed to store memory item: {type(e).__name__}")
            return False
    
    def search_items(
        self, 
        user_id: str, 
        query: str, 
        collection_name: str,
        item_type: Optional[str] = None,
        limit: int = 10
    ) -> List[MemoryItem]:
        """Search memory items by text query."""
        try:
            self.ensure_collection(collection_name)
            
            embedding = self._get_embedding(query)
            if embedding == [0.0] * self._vector_size:
                return []
            
            filter_conditions: List[Condition] = [
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            ]
            
            if item_type:
                filter_conditions.append(
                    FieldCondition(key="item_type", match=MatchValue(value=item_type))
                )
            
            search_result = self._client.search(
                collection_name=collection_name,
                query_vector=embedding,
                query_filter=Filter(must=filter_conditions),
                limit=limit,
                with_payload=True,
            )
            
            return self._points_to_memory_items(search_result)
            
        except Exception as e:
            logger.error(f"Failed to search memory items: {type(e).__name__}")
            return []
    
    def get_items(
        self,
        user_id: str,
        collection_name: str,
        item_type: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[MemoryItem]:
        """Retrieve memory items with optional filtering."""
        try:
            self.ensure_collection(collection_name)
            
            filter_conditions: List[Condition] = [
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            ]
            
            if item_type:
                filter_conditions.append(
                    FieldCondition(key="item_type", match=MatchValue(value=item_type))
                )
            
            if filters:
                for key, value in filters.items():
                    filter_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
            
            scroll_result = self._client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(must=filter_conditions),
                limit=limit,
                with_payload=True,
            )
            points, _ = scroll_result
            
            return self._points_to_memory_items(points)
            
        except Exception as e:
            logger.error(f"Failed to retrieve memory items: {type(e).__name__}")
            return []
    
    def update_item(
        self,
        item_id: str,
        user_id: str,
        collection_name: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update a memory item."""
        try:
            self.ensure_collection(collection_name)
            
            # Find the item first
            search_result = self._client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                        FieldCondition(key="memory_item_id", match=MatchValue(value=item_id)),
                    ]
                ),
                limit=1,
                with_payload=True,
            )
            
            points, _ = search_result
            if not points:
                return False
            
            point_id = points[0].id
            self._client.set_payload(
                collection_name=collection_name,
                points=[point_id],
                payload=updates,
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update memory item: {type(e).__name__}")
            return False
    
    def delete_item(self, item_id: str, user_id: str, collection_name: str) -> bool:
        """Delete a memory item."""
        try:
            self.ensure_collection(collection_name)
            
            # Find and delete the item
            search_result = self._client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                        FieldCondition(key="memory_item_id", match=MatchValue(value=item_id)),
                    ]
                ),
                limit=1,
                with_payload=True,
            )
            
            points, _ = search_result
            if not points:
                return False
            
            point_id = points[0].id
            self._client.delete(
                collection_name=collection_name,
                points_selector=[point_id],
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory item: {type(e).__name__}")
            return False
    
    def _points_to_memory_items(self, points) -> List[MemoryItem]:
        """Convert Qdrant points to MemoryItem objects."""
        memory_items = []
        for point in points:
            if point and hasattr(point, "payload") and point.payload:
                payload = point.payload
                memory_item = MemoryItem(
                    id=payload.get("memory_item_id", ""),
                    user_id=payload.get("user_id", ""),
                    content=payload.get("content", ""),
                    item_type=payload.get("item_type", ""),
                    metadata=payload.get("metadata", {}),
                    relevance_score=payload.get("relevance_score", 1.0),
                    tags=payload.get("tags", []),
                    created_at=payload.get("created_at", ""),
                    last_used=payload.get("last_used"),
                )
                memory_items.append(memory_item)
        return memory_items


class MemoryExtractor:
    """Extract structured information from text using LLM."""
    
    def __init__(self, config: MemoryConfig):
        self._config = config
        self._llm = init_chat_model(
            model=config.llm_model,
            model_provider="ollama",
            base_url=config.llm_base_url,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens
        )
    
    def extract_info(
        self, 
        text: str, 
        user_id: str,
        extraction_schema: Dict[str, Any],
        context: Optional[str] = None
    ) -> List[MemoryItem]:
        """Extract structured information from text based on schema."""
        
        available_types = extraction_schema.get("types", ["personal", "preference", "project"])
        available_tags = extraction_schema.get("tags", ["personal", "work", "hobby"])
        
        types_str = '", "'.join(available_types)
        tags_str = ', '.join(available_tags)
        
        context_prompt = f"\nContext: {context}" if context else ""
        
        extraction_prompt = f"""
        Analyze this text and extract meaningful information that would be useful for personalization and context.
        
        Text: "{text}"{context_prompt}
        
        Extract information that would help provide better, more personalized assistance.
        
        Return a JSON array of objects with:
        - item_type: one of ["{types_str}"]
        - content: detailed, context-rich description
        - relevance_score: 0.1-1.0 (how useful for personalization)
        - tags: relevant tags from [{tags_str}]
        - metadata: additional structured data as key-value pairs
        
        Only extract clear, factual information. Skip generic content.
        If no useful information, return empty array [].
        
        Input: "{text}"
        Output:
        """
        
        try:
            response = self._llm.invoke([{"role": "user", "content": extraction_prompt}])
            response_text = str(response.content if hasattr(response, "content") else response)
            
            # Extract JSON array from response
            json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                extracted_data = json.loads(json_str)
                
                memory_items = []
                for data in extracted_data:
                    if isinstance(data, dict) and data.get("content"):
                        memory_item = MemoryItem(
                            id=f"memory_{uuid.uuid4().hex[:8]}",
                            user_id=user_id,
                            content=data["content"],
                            item_type=data.get("item_type", "personal"),
                            metadata=data.get("metadata", {}),
                            relevance_score=float(data.get("relevance_score", 0.5)),
                            tags=data.get("tags", [])
                        )
                        memory_items.append(memory_item)
                
                return memory_items
        
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to extract information: {type(e).__name__}")
        
        return []


class MemoryManager:
    """High-level memory management interface."""
    
    def __init__(self, config: MemoryConfig, backend: Optional[MemoryBackend] = None):
        self._config = config
        self._backend = backend or QdrantMemoryBackend(config)
        self._extractor = MemoryExtractor(config)
        self._collections = {}  # Track registered collections
    
    def register_collection(self, name: str, description: str = "") -> None:
        """Register a memory collection."""
        self._collections[name] = description
        self._backend.ensure_collection(name)
    
    def store(self, item: MemoryItem, collection_name: str) -> bool:
        """Store a memory item."""
        return self._backend.store_item(item, collection_name)
    
    def search(
        self, 
        user_id: str, 
        query: str, 
        collection_name: str,
        item_type: Optional[str] = None,
        limit: int = 10
    ) -> List[MemoryItem]:
        """Search memory items."""
        return self._backend.search_items(user_id, query, collection_name, item_type, limit)
    
    def get_all(
        self,
        user_id: str,
        collection_name: str,
        item_type: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[MemoryItem]:
        """Get memory items with filtering."""
        return self._backend.get_items(user_id, collection_name, item_type, filters, limit)
    
    def update(
        self,
        item_id: str,
        user_id: str,
        collection_name: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update a memory item."""
        return self._backend.update_item(item_id, user_id, collection_name, updates)
    
    def delete(self, item_id: str, user_id: str, collection_name: str) -> bool:
        """Delete a memory item."""
        return self._backend.delete_item(item_id, user_id, collection_name)
    
    def extract_and_store(
        self,
        text: str,
        user_id: str,
        collection_name: str,
        extraction_schema: Dict[str, Any],
        context: Optional[str] = None,
        dedup_limit: int = 5
    ) -> List[MemoryItem]:
        """Extract information from text and store it, with deduplication."""
        extracted_items = self._extractor.extract_info(text, user_id, extraction_schema, context)
        
        stored_items = []
        for item in extracted_items:
            # Check for duplicates
            if dedup_limit > 0:
                existing_items = self.search(user_id, item.content, collection_name, limit=dedup_limit)
                
                # Simple similarity check
                is_duplicate = any(
                    existing.content.lower().strip() == item.content.lower().strip() or
                    (len(existing.content) > 10 and existing.content.lower() in item.content.lower()) or
                    (len(item.content) > 10 and item.content.lower() in existing.content.lower())
                    for existing in existing_items
                )
                
                if is_duplicate:
                    logger.info(f"Skipping duplicate memory item: '{item.content}'")
                    continue
            
            if self.store(item, collection_name):
                stored_items.append(item)
                logger.info(f"Stored memory item: '{item.content}'")
        
        return stored_items
    
    def get_context_summary(self, user_id: str, collection_name: str, limit: int = 20) -> str:
        """Get a summary of user context for a collection."""
        items = self.get_all(user_id, collection_name, limit=limit)
        
        if not items:
            return "No context available"
        
        # Group by type
        by_type = {}
        for item in items:
            if item.item_type not in by_type:
                by_type[item.item_type] = []
            by_type[item.item_type].append(item.content)
        
        # Create summary
        summary_parts = []
        for item_type, contents in by_type.items():
            if contents:
                summary_parts.append(f"{item_type.title()}: {'; '.join(contents[:3])}")
        
        return "; ".join(summary_parts) if summary_parts else "No context available" 