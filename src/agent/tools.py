"""Tools for todo management with unified storage and simplified architecture."""

from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import re
import logging
import uuid
from langchain_core.tools import BaseTool
from langchain.chat_models import init_chat_model
from mem0 import Memory
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
from src.agent.models import (
    TodoItem,
    UserInfo,
    UserContext,
    UserInfoType,
    TodoTag,
    UserInfoTag,
    CreateTodosInput,
    ListTodosInput,
    CompleteTodoInput,
    ExtractUserInfoInput,
)
from src.agent.config import AgentConfig
import requests

logger = logging.getLogger(__name__)


def ensure_qdrant_collections(config: AgentConfig) -> None:
    """Ensure both todo and user profile collections exist in Qdrant."""
    try:
        # Initialize Qdrant client
        if config.qdrant.url:
            client = QdrantClient(
                url=config.qdrant.url,
                api_key=config.qdrant.api_key
            )
        else:
            client = QdrantClient(
                host=config.qdrant.host,
                port=config.qdrant.port
            )
        
        # Collection configuration
        vector_config = VectorParams(
            size=config.qdrant.vector_size,
            distance=Distance.COSINE
        )
        
        # Ensure todo collection exists
        try:
            client.get_collection(config.qdrant.todo_collection_name)
            logger.info(f"Todo collection '{config.qdrant.todo_collection_name}' already exists")
        except Exception:
            client.create_collection(
                collection_name=config.qdrant.todo_collection_name,
                vectors_config=vector_config
            )
            logger.info(f"Created todo collection '{config.qdrant.todo_collection_name}'")
        
        # Ensure user profile collection exists
        try:
            client.get_collection(config.qdrant.user_profile_collection_name)
            logger.info(f"User profile collection '{config.qdrant.user_profile_collection_name}' already exists")
        except Exception:
            client.create_collection(
                collection_name=config.qdrant.user_profile_collection_name,
                vectors_config=vector_config
            )
            logger.info(f"Created user profile collection '{config.qdrant.user_profile_collection_name}'")
            
    except Exception as e:
        logger.error(f"Failed to ensure Qdrant collections: {type(e).__name__}")


class UnifiedTodoStorage:
    """Unified storage handler for all todo operations using Qdrant only."""

    def __init__(self, config: AgentConfig):
        self._config = config
        
        # Ensure collections exist
        ensure_qdrant_collections(config)
        
        # Initialize Qdrant client with config values
        if config.qdrant.url:
            # Use cloud/remote URL if available
            self._qdrant_client = QdrantClient(
                url=config.qdrant.url,
                api_key=config.qdrant.api_key
            )
        else:
            # Use local host/port
            self._qdrant_client = QdrantClient(
                host=config.qdrant.host,
                port=config.qdrant.port
            )
        self._embedding_url = f"{config.ollama.base_url}/api/embeddings"
        self._collection_name = config.qdrant.todo_collection_name
        self._embedding_model = "bge-m3:latest"  # From embedder config
        self._vector_size = config.qdrant.vector_size

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Ollama."""
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

    def store_todo(self, todo: TodoItem, user_id: str) -> bool:
        """Store a todo item in Qdrant using legacy-compatible structure."""
        try:
            content = f"Todo: {todo.title} - {todo.description} - Priority: {todo.priority} - Tags: {','.join(todo.tags)}"
            embedding = self._get_embedding(content)

            # Generate UUID for point ID (Qdrant requirement)
            point_uuid = str(uuid.uuid4())

            # Use legacy-compatible payload structure
            payload = {
                "type": "todo_item_direct",  # Match existing type
                "user_id": user_id,
                "todo_id": todo.id,  # Use separate field for todo ID
                "title": todo.title,
                "description": todo.description,
                "priority": todo.priority,
                "due_date": todo.due_date,
                "completed": todo.completed,
                "completed_at": todo.completed_at,
                "created_at": todo.created_at,
                "tags": todo.tags,
                "content": content,  # For searchability
            }

            point = PointStruct(
                id=point_uuid,  # Use UUID for point ID
                vector=embedding,
                payload=payload,
            )

            try:
                result = self._qdrant_client.upsert(
                    collection_name=self._collection_name, points=[point]
                )
                return result.status == UpdateStatus.COMPLETED
            except Exception as upsert_error:
                logger.warning(f"Qdrant upsert failed: {type(upsert_error).__name__}")
                return False

        except Exception as e:
            logger.error(f"Failed to store todo: {type(e).__name__}")
            return False

    def complete_todo(self, todo_id: str, user_id: str) -> bool:
        """Mark a todo as completed by updating its payload."""
        try:
            search_result = self._qdrant_client.scroll(
                collection_name=self._collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                        FieldCondition(key="todo_id", match=MatchValue(value=todo_id)),
                        FieldCondition(key="type", match=MatchValue(value="todo_item_direct")),
                    ]
                ),
                limit=1,
                with_payload=True,
            )

            points, _ = search_result
            if not points:
                logger.warning("Todo not found for completion request")
                return False

            # Update the payload to mark as completed
            completion_time = datetime.now().isoformat()

            # Use the actual Qdrant point ID, not the todo_id
            point_id = points[0].id
            self._qdrant_client.set_payload(
                collection_name=self._collection_name,
                points=[point_id],
                payload={"completed": True, "completed_at": completion_time},
            )

            return True

        except Exception as e:
            logger.error(f"Failed to complete todo: {type(e).__name__}")
            return False

    def get_todos(
        self,
        user_id: str,
        completed: Optional[bool] = None,
        priority: Optional[str] = None,
        search_query: Optional[str] = None,
        limit: int = 100,
    ) -> List[TodoItem]:
        """Retrieve todos with optional filtering."""
        try:
            # Build filter conditions - use existing field names from legacy storage
            filter_conditions: List[Condition] = [
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(
                    key="type", match=MatchValue(value="todo_item_direct")
                ),  # Use existing type
            ]

            if completed is not None:
                filter_conditions.append(
                    FieldCondition(key="completed", match=MatchValue(value=completed))
                )

            if priority:
                filter_conditions.append(
                    FieldCondition(key="priority", match=MatchValue(value=priority))
                )

            # Perform the search with error handling
            points = []
            try:
                if search_query:
                    # Use vector search for text queries
                    embedding = self._get_embedding(search_query)
                    if (
                        embedding != [0.0] * self._vector_size
                    ):  # Only search if we have valid embedding
                        search_result = self._qdrant_client.search(
                            collection_name=self._collection_name,
                            query_vector=embedding,
                            query_filter=Filter(must=filter_conditions),
                            limit=limit,
                            with_payload=True,
                        )
                        points = search_result
                else:
                    # Use scroll for filtering without text search
                    scroll_result = self._qdrant_client.scroll(
                        collection_name=self._collection_name,
                        scroll_filter=Filter(must=filter_conditions),
                        limit=limit,
                        with_payload=True,
                    )
                    points, _ = scroll_result
            except Exception as query_error:
                logger.warning(
                    f"Qdrant query failed, returning empty results: {type(query_error).__name__}"
                )
                return []

            todos = []
            for point in points:
                if point and hasattr(point, "payload") and point.payload:
                    payload = point.payload
                    todo = TodoItem(
                        id=payload.get("todo_id", payload.get("id", "")),
                        title=payload.get("title", ""),
                        description=payload.get(
                            "description", payload.get("title", "")
                        ),
                        priority=payload.get("priority", "medium"),
                        due_date=payload.get("due_date"),
                        completed=payload.get("completed", False),
                        completed_at=payload.get("completed_at"),
                        created_at=payload.get("created_at", ""),
                        tags=payload.get("tags", []),
                    )
                    todos.append(todo)

            return todos

        except Exception as e:
            logger.error(f"Failed to retrieve todos: {type(e).__name__}")
            return []


class UserInfoStorage:
    """Storage handler for user information and context using Qdrant."""
    
    def __init__(self, config: AgentConfig):
        self._config = config
        
        # Ensure collections exist  
        ensure_qdrant_collections(config)
        
        # Initialize Qdrant client with config values
        if config.qdrant.url:
            self._qdrant_client = QdrantClient(
                url=config.qdrant.url,
                api_key=config.qdrant.api_key
            )
        else:
            self._qdrant_client = QdrantClient(
                host=config.qdrant.host,
                port=config.qdrant.port
            )
        self._embedding_url = f"{config.ollama.base_url}/api/embeddings"
        self._collection_name = config.qdrant.user_profile_collection_name
        self._embedding_model = "bge-m3:latest"
        self._vector_size = config.qdrant.vector_size
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Ollama."""
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
    
    def store_user_info(self, user_info: UserInfo) -> bool:
        """Store user information in Qdrant with deduplication."""
        try:
            # Check for existing similar content to avoid duplicates
            existing_infos = self.search_user_info(
                user_info.user_id, 
                user_info.content, 
                limit=5
            )
            
            # Check if we already have very similar content
            for existing in existing_infos:
                # Simple similarity check - if content is very similar, skip storing
                if (existing.content.lower().strip() == user_info.content.lower().strip() or
                    (len(existing.content) > 10 and existing.content.lower() in user_info.content.lower()) or
                    (len(user_info.content) > 10 and user_info.content.lower() in existing.content.lower())):
                    
                    logger.info(f"Skipping duplicate user info: '{user_info.content}' (similar to existing: '{existing.content}')")
                    return True  # Return True since the info is already stored
            
            # Create searchable content
            content = f"User info ({user_info.info_type}): {user_info.content}"
            if user_info.tags:
                content += f" - Tags: {', '.join(user_info.tags)}"
            
            embedding = self._get_embedding(content)
            point_uuid = str(uuid.uuid4())
            
            payload = {
                "type": "user_info",
                "user_id": user_info.user_id,
                "info_id": user_info.id,
                "info_type": user_info.info_type,
                "content": user_info.content,
                "relevance_score": user_info.relevance_score,
                "tags": user_info.tags,
                "created_at": user_info.created_at,
                "last_used": user_info.last_used,
                "searchable_content": content,
            }
            
            point = PointStruct(
                id=point_uuid,
                vector=embedding,
                payload=payload,
            )
            
            result = self._qdrant_client.upsert(
                collection_name=self._collection_name, points=[point]
            )
            
            if result.status == UpdateStatus.COMPLETED:
                logger.info(f"Stored new user info: '{user_info.content}'")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to store user info: {type(e).__name__}")
            return False
    
    def get_user_context(self, user_id: str, limit: int = 20) -> UserContext:
        """Retrieve user context for todo creation."""
        try:
            # Get all user info for this user
            filter_conditions: List[Condition] = [
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="type", match=MatchValue(value="user_info")),
            ]
            
            scroll_result = self._qdrant_client.scroll(
                collection_name=self._collection_name,
                scroll_filter=Filter(must=filter_conditions),
                limit=limit,
                with_payload=True,
            )
            points, _ = scroll_result
            
            # Organize by info type
            user_context = UserContext(user_id=user_id)
            
            for point in points:
                if point and hasattr(point, "payload") and point.payload:
                    payload = point.payload
                    
                    # Parse info_type from string to enum
                    info_type_str = payload.get("info_type", "personal")
                    try:
                        info_type = UserInfoType(info_type_str)
                    except ValueError:
                        info_type = UserInfoType.PERSONAL
                    
                    user_info = UserInfo(
                        id=payload.get("info_id", ""),
                        user_id=payload.get("user_id", user_id),
                        info_type=info_type,
                        content=payload.get("content", ""),
                        relevance_score=payload.get("relevance_score", 1.0),
                        tags=payload.get("tags", []),
                        created_at=payload.get("created_at", ""),
                        last_used=payload.get("last_used"),
                    )
                    
                    # Add to context using the new method
                    user_context.add_info(user_info)
            
            return user_context
            
        except Exception as e:
            logger.error(f"Failed to retrieve user context: {type(e).__name__}")
            return UserContext(user_id=user_id)
    
    def search_user_info(self, user_id: str, query: str, limit: int = 10) -> List[UserInfo]:
        """Search user information by text query."""
        try:
            embedding = self._get_embedding(query)
            if embedding == [0.0] * self._vector_size:
                return []
            
            filter_conditions: List[Condition] = [
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="type", match=MatchValue(value="user_info")),
            ]
            
            search_result = self._qdrant_client.search(
                collection_name=self._collection_name,
                query_vector=embedding,
                query_filter=Filter(must=filter_conditions),
                limit=limit,
                with_payload=True,
            )
            
            user_infos = []
            for point in search_result:
                if point and hasattr(point, "payload") and point.payload:
                    payload = point.payload
                    
                    # Parse info_type from string to enum
                    info_type_str = payload.get("info_type", "personal")
                    try:
                        info_type = UserInfoType(info_type_str)
                    except ValueError:
                        info_type = UserInfoType.PERSONAL
                    
                    user_info = UserInfo(
                        id=payload.get("info_id", ""),
                        user_id=payload.get("user_id", user_id),
                        info_type=info_type,
                        content=payload.get("content", ""),
                        relevance_score=payload.get("relevance_score", 1.0),
                        tags=payload.get("tags", []),
                        created_at=payload.get("created_at", ""),
                        last_used=payload.get("last_used"),
                    )
                    user_infos.append(user_info)
            
            return user_infos
            
        except Exception as e:
            logger.error(f"Failed to search user info: {type(e).__name__}")
            return []


class TodoManagerTool(BaseTool):
    """Simplified todo creation tool with single LLM call and unified storage."""

    name: str = "todo_manager"
    description: str = """Create todo items from natural language with reliable single-pass extraction.

    Efficiently handles:
    - Multiple tasks from one request
    - Priority and due date detection
    - Tag categorization
    - Unified storage in Qdrant
    
    Use for: "Add X", "Remind me to Y", "I need to do Z by tomorrow"
    """
    args_schema: type = CreateTodosInput

    def __init__(self, memory: Memory, config: AgentConfig, **kwargs):
        super().__init__(**kwargs)
        self._memory = memory
        self._config = config
        self._storage = UnifiedTodoStorage(config)
        self._llm = init_chat_model(
            model=config.ollama.model, 
            model_provider="ollama",
            base_url=config.ollama.base_url,
            temperature=config.ollama.temperature,
            max_tokens=config.ollama.max_tokens
        )

    def _extract_todos_structured(self, user_input: str) -> List[Dict[str, Any]]:
        """Extract todos using single structured LLM call."""
        # Get available todo tags from enum
        available_tags = TodoTag.get_all_tags()
        tags_str = ', '.join(available_tags)
        
        prompt = f"""Extract todo items from: "{user_input}"

Return valid JSON array of objects. Each object must have:
- title: string (required, max 100 chars)
- priority: "high", "medium", or "low" 
- due_date: string or null (tomorrow, today, Friday, etc.)
- tags: array of strings from [{tags_str}]

Examples:
Input: "buy groceries and call mom tomorrow"
Output: [
  {{"title": "buy groceries", "priority": "medium", "due_date": "tomorrow", "tags": ["shopping"]}},
  {{"title": "call mom", "priority": "medium", "due_date": "tomorrow", "tags": ["personal"]}}
]

Input: "urgent: fix the bug"
Output: [
  {{"title": "fix the bug", "priority": "high", "due_date": null, "tags": ["work"]}}
]

Input: "{user_input}"
Output:"""

        try:
            response = self._llm.invoke([{"role": "user", "content": prompt}])
            response_text = str(
                response.content if hasattr(response, "content") else response
            )

            # Extract JSON from response
            json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                todos = json.loads(json_str)

                # Validate and clean each todo
                cleaned_todos = []
                for todo in todos:
                    if isinstance(todo, dict) and todo.get("title"):
                        cleaned_todo = {
                            "title": str(todo["title"])[:100].strip(),
                            "priority": todo.get("priority", "medium").lower(),
                            "due_date": todo.get("due_date"),
                            "tags": todo.get("tags", []),
                        }

                        # Validate priority
                        if cleaned_todo["priority"] not in ["high", "medium", "low"]:
                            cleaned_todo["priority"] = "medium"

                        # Validate tags using enum
                        valid_tags = set(TodoTag.get_all_tags())
                        cleaned_todo["tags"] = [
                            tag for tag in cleaned_todo["tags"] if tag in valid_tags
                        ]

                        cleaned_todos.append(cleaned_todo)

                return cleaned_todos

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(
                f"Failed to parse structured LLM response: {type(e).__name__}"
            )

        # Fallback to simple parsing
        return self._fallback_parse(user_input)

    def _fallback_parse(self, user_input: str) -> List[Dict[str, Any]]:
        """Simple fallback parsing when structured extraction fails."""
        # Simple split on common separators
        text = re.sub(
            r"(?:i need to|remind me to|add|create)\s+",
            "",
            user_input,
            flags=re.IGNORECASE,
        ).strip()
        parts = re.split(r",\s*(?:and\s+)?|\s+and\s+|\d+\.\s*", text)

        todos = []
        for part in parts:
            title = part.strip().strip(".,")
            if title and len(title) > 2:
                # Simple priority detection
                priority = (
                    "high"
                    if any(
                        word in title.lower() for word in ["urgent", "asap", "critical"]
                    )
                    else "medium"
                )

                # Simple due date detection
                due_date = None
                if any(
                    word in title.lower() for word in ["tomorrow", "today", "tonight"]
                ):
                    due_date = "tomorrow" if "tomorrow" in title.lower() else "today"

                todos.append(
                    {
                        "title": title[:100],
                        "priority": priority,
                        "due_date": due_date,
                        "tags": [],
                    }
                )

        # If no todos extracted, create one from the entire input
        if not todos:
            todos = [
                {
                    "title": user_input.strip()[:100],
                    "priority": "medium",
                    "due_date": None,
                    "tags": [],
                }
            ]

        return todos

    def _run(self, user_input: str, user_id: str) -> str:
        """Execute the tool to create todo items with simplified single-call extraction."""
        try:
            # Single structured extraction call
            todo_data = self._extract_todos_structured(user_input)
            if not todo_data:
                return json.dumps(
                    {
                        "status": "no_todos_extracted",
                        "message": "No actionable todo items found in your request",
                        "user_input": user_input,
                    }
                )

            # Create TodoItem objects and store them
            created_todos = []
            failed_todos = []

            for data in todo_data:
                todo_id = f"todo_{uuid.uuid4().hex[:8]}"
                todo = TodoItem(
                    id=todo_id,
                    title=data["title"],
                    description=data[
                        "title"
                    ],  # Use title as description for simplicity
                    priority=data["priority"],
                    due_date=data["due_date"],
                    tags=data["tags"],
                )

                # Store in unified storage
                if self._storage.store_todo(todo, user_id):
                    created_todos.append(todo.to_dict())
                else:
                    failed_todos.append(data["title"])

            if not created_todos:
                return json.dumps(
                    {
                        "status": "storage_failed",
                        "message": "Failed to store any todo items",
                        "user_input": user_input,
                        "failed_todos": failed_todos,
                    }
                )

            # Compute summary statistics
            priority_counts = {"high": 0, "medium": 0, "low": 0}
            due_date_count = 0
            all_tags = set()

            for todo in created_todos:
                priority_counts[todo["priority"]] += 1
                if todo["due_date"]:
                    due_date_count += 1
                all_tags.update(todo["tags"])

            # Return structured data
            result_data = {
                "status": "success",
                "action": "todos_created",
                "user_id": user_id,
                "user_input": user_input,
                "created_count": len(created_todos),
                "failed_count": len(failed_todos),
                "todos": created_todos,
                "failed_todos": failed_todos,
                "summary": {
                    "total": len(created_todos),
                    "priorities": priority_counts,
                    "with_due_dates": due_date_count,
                    "tags": list(all_tags),
                },
            }

            return json.dumps(result_data, indent=2)

        except Exception as e:
            logger.error(f"Error in todo creation: {type(e).__name__}")
            return json.dumps(
                {
                    "status": "error",
                    "action": "todo_creation_failed",
                    "user_id": user_id,
                    "error": "Internal processing error",
                    "suggestion": "Please try again with a simpler request.",
                },
                indent=2,
            )


class ListTodosTool(BaseTool):
    """Simplified todo listing tool with unified storage and database-level filtering."""

    name: str = "list_todos"
    description: str = """List and filter todo items with efficient database queries.

    Efficiently handles:
    - Retrieving todos with database-level filtering
    - Completion status, priority, and text search
    - Organized results with statistics
    
    Use for: "Show my todos", "Find work tasks", "What's completed?"
    """
    args_schema: type = ListTodosInput

    def __init__(self, memory: Memory, config: AgentConfig, **kwargs):
        super().__init__(**kwargs)
        self._memory = memory
        self._config = config
        self._storage = UnifiedTodoStorage(config)

    @property
    def memory(self) -> Memory:
        """Access the memory instance."""
        if not hasattr(self, "_memory") or self._memory is None:
            raise RuntimeError("Memory not properly initialized in ListTodosTool")
        return self._memory

    def _run(
        self,
        user_id: str,
        filter_completed: Optional[str] = None,
        filter_priority: Optional[str] = None,
        search_query: Optional[str] = None,
    ) -> str:
        """List and filter todo items using unified storage."""
        try:
            # Parse completion filter with proper validation
            completed_filter: Optional[bool] = None
            if filter_completed and filter_completed.lower().strip() not in [
                "null",
                "none",
                "",
            ]:
                filter_completed_lower = filter_completed.lower().strip()
                if filter_completed_lower in ["true", "1", "yes", "on"]:
                    completed_filter = True
                elif filter_completed_lower in ["false", "0", "no", "off"]:
                    completed_filter = False

            # Use unified storage to retrieve todos with database-level filtering
            todos = self._storage.get_todos(
                user_id=user_id,
                completed=completed_filter,
                priority=filter_priority,
                search_query=search_query,
                limit=100,
            )

            if not todos:
                return json.dumps(
                    {
                        "status": "no_todos_found",
                        "message": f"No todos found for user {user_id}",
                        "user_id": user_id,
                        "search_query": search_query,
                        "filters": {
                            "completed": completed_filter,
                            "priority": filter_priority,
                        },
                        "suggestion": "Try creating some todos first!",
                    }
                )

            # Convert TodoItem objects to dictionaries and categorize
            todo_dicts = [todo.to_dict() for todo in todos]
            pending_todos = [t for t in todo_dicts if not t["completed"]]
            completed_todos = [t for t in todo_dicts if t["completed"]]

            result_data = {
                "status": "success",
                "user_id": user_id,
                "search_query": search_query,
                "total_todos": len(todo_dicts),
                "pending_count": len(pending_todos),
                "completed_count": len(completed_todos),
                "filters": {"completed": completed_filter, "priority": filter_priority},
                "todos": {"pending": pending_todos, "completed": completed_todos},
            }

            return json.dumps(result_data, indent=2)

        except Exception as e:
            logger.error(f"Error in todo retrieval: {type(e).__name__}")
            return json.dumps(
                {
                    "status": "error",
                    "action": "todo_retrieval_failed",
                    "user_id": user_id,
                    "error": "Internal retrieval error",
                    "error_type": type(e).__name__,
                },
                indent=2,
            )


class CompleteTodoTool(BaseTool):
    """Tool for marking todo items as complete using unified storage."""

    name: str = "complete_todo"
    description: str = "Mark a todo item as completed with atomic updates."
    args_schema: type = CompleteTodoInput

    def __init__(self, memory: Memory, config: AgentConfig, **kwargs):
        super().__init__(**kwargs)
        self._memory = memory
        self._config = config
        self._storage = UnifiedTodoStorage(config)

    def _run(self, todo_id: str, user_id: str) -> str:
        """Mark a todo as complete using unified storage."""
        try:
            # Use unified storage to complete the todo atomically
            success = self._storage.complete_todo(todo_id, user_id)

            if not success:
                return json.dumps(
                    {
                        "status": "not_found",
                        "action": "todo_completion_failed",
                        "todo_id": todo_id,
                        "user_id": user_id,
                        "error": "Todo not found or does not belong to user",
                    },
                    indent=2,
                )

            # Return structured data
            result_data = {
                "status": "success",
                "action": "todo_completed",
                "todo_id": todo_id,
                "user_id": user_id,
                "completed_at": datetime.now().isoformat(),
            }

            return json.dumps(result_data, indent=2)

        except Exception as e:
            logger.error(f"Error in todo completion: {type(e).__name__}")
            return json.dumps(
                {
                    "status": "error",
                    "action": "todo_completion_failed",
                    "todo_id": todo_id,
                    "user_id": user_id,
                    "error": "Internal completion error",
                },
                indent=2,
            )


class UserInfoExtractor:
    """Extract and store user information from conversations."""
    
    def __init__(self, config: AgentConfig):
        self._config = config
        self._storage = UserInfoStorage(config)
        self._llm = init_chat_model(
            model=config.ollama.model,
            model_provider="ollama",
            base_url=config.ollama.base_url,
            temperature=0.1,
            max_tokens=config.ollama.max_tokens
        )
    
    def extract_and_store_user_info(self, conversation_text: str, user_id: str) -> List[UserInfo]:
        """Extract user information from conversation and store it."""
        
        # Get available info types and tags from enums
        available_types = UserInfoType.get_all_types()
        types_str = '", "'.join(available_types)
        
        available_tags = UserInfoTag.get_all_tags()
        tags_str = ', '.join(available_tags)
        
        extraction_prompt = f"""
        You are Navis, a todo management assistant. Analyze this conversation and extract rich, contextual personal information about the user that will help you provide better, more personalized assistance in future interactions.
        
        Conversation: "{conversation_text}"
        
        Extract detailed, context-rich information that would be helpful for:
        - Creating more relevant and personalized todos
        - Understanding the user's work context and responsibilities
        - Adapting to their preferences and communication style
        - Providing better suggestions and reminders
        
        Focus on extracting:
        - Personal details (full name, job title/role, company, family members, location)
        - Work context (current projects, responsibilities, team, schedule, deadlines)
        - Preferences (working hours, communication style, priorities, tools they use)
        - Interests and hobbies (activities they enjoy, skills, passions)
        - Goals and aspirations (career goals, personal objectives, learning interests)
        - Routine and habits (daily patterns, meeting schedules, recurring activities)
        
        IMPORTANT: Make each extracted piece of information rich and specific. Instead of "works at Google", extract "works as a Senior Data Scientist at Google focusing on machine learning for search algorithms".
        
        Return a JSON array of objects with:
        - info_type: one of ["{types_str}"]
        - content: detailed, context-rich description of the information
        - relevance_score: 0.1-1.0 (how useful for creating personalized todos)
        - tags: relevant tags from [{tags_str}]
        
        Only extract clear, factual information. Skip generic greetings.
        If no useful information, return empty array [].
        
        Example:
        Input: "Hi, I'm Sarah Chen and I work as a Senior Data Scientist at Google in the Search team. I'm currently working on improving recommendation algorithms and I have a big presentation next Friday. I usually work best in the mornings and I love rock climbing on weekends."
        Output: [
          {{"info_type": "personal", "content": "full name is Sarah Chen", "relevance_score": 1.0, "tags": ["personal"]}},
          {{"info_type": "personal", "content": "works as Senior Data Scientist at Google Search team, focuses on recommendation algorithms", "relevance_score": 0.95, "tags": ["work"]}},
          {{"info_type": "project", "content": "currently working on improving recommendation algorithms with upcoming presentation next Friday", "relevance_score": 0.9, "tags": ["work", "deadline"]}},
          {{"info_type": "preference", "content": "works best and is most productive during morning hours", "relevance_score": 0.8, "tags": ["work", "schedule"]}},
          {{"info_type": "personal", "content": "enjoys rock climbing as weekend hobby and stress relief activity", "relevance_score": 0.7, "tags": ["hobby", "schedule"]}}
        ]
        
        Input: "{conversation_text}"
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
                
                stored_infos = []
                for data in extracted_data:
                    if isinstance(data, dict) and data.get("content"):
                        # Parse info_type from string to enum
                        info_type_str = data.get("info_type", "personal")
                        try:
                            info_type = UserInfoType(info_type_str)
                        except ValueError:
                            # Default to personal if invalid type
                            info_type = UserInfoType.PERSONAL
                        
                        # Validate and filter tags using enum
                        raw_tags = data.get("tags", [])
                        valid_user_tags = set(UserInfoTag.get_all_tags())
                        validated_tags = [tag for tag in raw_tags if tag in valid_user_tags]
                        
                        # Create UserInfo object
                        user_info = UserInfo(
                            id=f"info_{uuid.uuid4().hex[:8]}",
                            user_id=user_id,
                            info_type=info_type,
                            content=data["content"],
                            relevance_score=float(data.get("relevance_score", 0.5)),
                            tags=validated_tags
                        )
                        
                        # Store the information
                        if self._storage.store_user_info(user_info):
                            stored_infos.append(user_info)
                            logger.info(f"Stored user info: {user_info.content}")
                
                return stored_infos
        
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to extract user info: {type(e).__name__}")
        
        return []


class UserMemorySearchTool(BaseTool):
    """Tool for searching user memories and personal information."""

    name: str = "search_user_memories"
    description: str = """ALWAYS use this tool when the user asks ANY personal question about themselves.

    REQUIRED for questions like:
    - "What is my name?" or "Who am I?" → search with "name"
    - "What do I do for work?" or "What's my job?" → search with "work job"
    - "What are my preferences?" → search with "preferences"
    - "Tell me about myself" → search with "personal information"
    - "What projects am I working on?" → search with "projects"
    - "What are my hobbies?" → search with "hobbies interests"
    
    NEVER answer personal questions without using this tool first. The user expects you to remember their information.
    """
    args_schema: type = ExtractUserInfoInput

    def __init__(self, memory: Memory, config: AgentConfig, **kwargs):
        super().__init__(**kwargs)
        self._memory = memory
        self._config = config
        self._storage = UserInfoStorage(config)

    def _run(self, query: str, user_id: str) -> str:
        """Search user memories and return relevant information."""
        try:
            # Search for relevant user information
            user_infos = self._storage.search_user_info(user_id, query, limit=10)
            
            if not user_infos:
                return json.dumps({
                    "status": "no_memories_found",
                    "message": "No relevant personal information found in your memories",
                    "query": query,
                    "user_id": user_id,
                    "suggestion": "You can share personal information with me and I'll remember it for future conversations."
                })
            
            # Organize results by info type
            personal_info = []
            preferences = []
            projects = []
            other_info = []
            
            for info in user_infos:
                info_dict = {
                    "content": info.content,
                    "relevance_score": info.relevance_score,
                    "tags": info.tags,
                    "created_at": info.created_at
                }
                
                if info.info_type == UserInfoType.PERSONAL:
                    personal_info.append(info_dict)
                elif info.info_type == UserInfoType.PREFERENCE:
                    preferences.append(info_dict)
                elif info.info_type == UserInfoType.PROJECT:
                    projects.append(info_dict)
                else:
                    other_info.append(info_dict)
            
            result_data = {
                "status": "success",
                "query": query,
                "user_id": user_id,
                "total_memories": len(user_infos),
                "memories": {
                    "personal": personal_info,
                    "preferences": preferences,
                    "projects": projects,
                    "other": other_info
                }
            }
            
            return json.dumps(result_data, indent=2)
            
        except Exception as e:
            logger.error(f"Error searching user memories: {type(e).__name__}")
            return json.dumps({
                "status": "error",
                "action": "memory_search_failed",
                "query": query,
                "user_id": user_id,
                "error": "Internal search error"
            })


def create_todo_tools(memory: Memory, config: AgentConfig) -> List[BaseTool]:
    """Create all todo management tools."""
    return [
        TodoManagerTool(memory=memory, config=config),
        ListTodosTool(memory=memory, config=config),
        CompleteTodoTool(memory=memory, config=config),
    ]


def create_conversation_tools(memory: Memory, config: AgentConfig) -> List[BaseTool]:
    """Create tools for conversation chatbot including memory search."""
    return [
        UserMemorySearchTool(memory=memory, config=config),
    ]
