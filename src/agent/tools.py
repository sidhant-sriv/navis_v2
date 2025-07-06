"""Tools for todo management with memory integration and improved system prompts."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import re
import logging
import uuid
import time
from langchain_core.tools import BaseTool
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from mem0 import Memory
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
import requests

# Set up logger for this module
logger = logging.getLogger(__name__)


@dataclass
class TodoItem:
    """A single todo item with structured data."""

    id: str
    title: str
    description: str
    priority: str = "medium"  # low, medium, high
    due_date: Optional[str] = None
    completed: bool = False
    created_at: str = ""
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class CreateTodosInput(BaseModel):
    """Input for creating multiple todo items."""

    user_input: str = Field(
        description="The user's natural language request to create todo items. Can contain multiple tasks."
    )
    user_id: str = Field(description="Unique identifier for the user making the request")


class TodoManagerTool(BaseTool):
    """Advanced todo creation and management tool with intelligent parsing using multiple LLM queries."""

    name: str = "todo_manager"
    description: str = """Create and manage multiple todo items from natural language input with intelligent parsing.

    This tool excels at:
    - Extracting multiple tasks from complex requests  
    - Auto-detecting priorities and due dates
    - Categorizing todos with relevant tags
    - Storing in memory for future access
    
    Use this tool when users want to:
    - Create new todo items ("Add X to my list")
    - Set reminders ("Remind me to X")  
    - Schedule tasks ("I need to do X by Y")
    - Organize multiple items ("Create todos for: 1. X 2. Y")
    """
    args_schema: type = CreateTodosInput

    def __init__(self, memory: Memory, **kwargs):
        super().__init__(**kwargs)
        self._memory = memory
        # Initialize LLM for multi-step extraction
        self._llm = init_chat_model(model="llama3.1:8b-instruct-q8_0", model_provider="ollama")
        
        # Initialize direct Qdrant client for bypassing Mem0
        self._qdrant_client = QdrantClient(host="localhost", port=6333)
        
        # Initialize embedding model for manual embeddings
        self._embedding_url = "http://localhost:11434/api/embeddings"

    @property
    def memory(self) -> Memory:
        """Access the memory instance."""
        if not hasattr(self, '_memory') or self._memory is None:
            raise RuntimeError("Memory not properly initialized in TodoManagerTool")
        return self._memory

    def _clean_llm_response(self, response_text: str) -> List[str]:
        """Clean LLM response by removing headers, meta-text, and explanations."""
        lines = response_text.strip().split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Skip common LLM headers and meta-text
            skip_patterns = [
                r'^here\s+are?\s+',
                r'^here\s+is\s+',
                r'^the\s+following\s+',
                r'^extracted?\s+',
                r'^tasks?:?\s*$',
                r'^priorities?:?\s*$',
                r'^due\s+dates?:?\s*$',
                r'^tags?:?\s*$',
                r'^timeframes?:?\s*$',
                r'^\d+\.\s*$',  # Just numbers like "1."
                r'^output:?\s*$',
                r'^result:?\s*$',
                r'^answer:?\s*$',
                r'actionable\s+tasks?',
                r'extracted?\s+tasks?',
                r'individual\s+tasks?',
                r'task\s+extract',
                r'extract.*task',
            ]
            
            # Check if line matches any skip pattern
            should_skip = any(re.match(pattern, line.lower()) for pattern in skip_patterns)
            
            if not should_skip and len(line) > 1:
                cleaned_lines.append(line)
        
        return cleaned_lines

    def _extract_tasks_llm(self, text: str) -> List[str]:
        """Extract individual tasks using LLM."""
        prompt = f"""Extract actionable tasks from: "{text}"

Return ONLY the task text, one per line. NO headers or explanations.

Examples:
Input: "buy groceries and call mom" 
Output:
buy groceries
call mom"""

        response = self._llm.invoke([{"role": "user", "content": prompt}])
        response_text = response.content if hasattr(response, 'content') else str(response)
        if not isinstance(response_text, str):
            response_text = str(response_text)
        
        # Clean the response
        tasks = self._clean_llm_response(response_text)
        
        # Fallback: simple split if LLM didn't work properly
        if not tasks:
            text_clean = re.sub(r'(?:i need to|remind me to|add|create)\s+', '', text, flags=re.IGNORECASE).strip()
            parts = re.split(r',\s*(?:and\s+)?|\s+and\s+|\d+\.\s*', text_clean)
            for part in parts:
                clean_part = part.strip().strip('.,')
                if clean_part and len(clean_part) > 2:
                    tasks.append(clean_part)
            if not tasks:
                tasks = [text.strip()]
        
        return tasks

    def _extract_priorities_llm(self, tasks: List[str]) -> List[str]:
        """Extract priority for each task using LLM."""
        tasks_text = '\n'.join(f"{i+1}. {task}" for i, task in enumerate(tasks))
        
        prompt = f"""For each task, return ONLY: high, medium, or low

Tasks:
{tasks_text}

Rules:
- HIGH: urgent, ASAP, critical, emergency
- LOW: later, eventually, when possible
- MEDIUM: default

Return one word per line:"""

        response = self._llm.invoke([{"role": "user", "content": prompt}])
        response_text = response.content if hasattr(response, 'content') else str(response)
        if not isinstance(response_text, str):
            response_text = str(response_text)
        
        # Clean and parse priorities
        lines = self._clean_llm_response(response_text)
        priorities = []
        for line in lines:
            priority = line.strip().lower()
            if priority in ['high', 'medium', 'low']:
                priorities.append(priority)
        
        # Fill missing with medium
        while len(priorities) < len(tasks):
            priorities.append('medium')
            
        return priorities[:len(tasks)]

    def _extract_due_dates_llm(self, tasks: List[str]) -> List[Optional[str]]:
        """Extract due dates for each task using LLM."""
        tasks_text = '\n'.join(f"{i+1}. {task}" for i, task in enumerate(tasks))
        
        prompt = f"""For each task, return timeframe or "none"

Tasks:
{tasks_text}

Look for: tomorrow, today, Friday, next week, ASAP

Return one per line:"""

        response = self._llm.invoke([{"role": "user", "content": prompt}])
        response_text = response.content if hasattr(response, 'content') else str(response)
        if not isinstance(response_text, str):
            response_text = str(response_text)
        
        # Clean and parse due dates
        lines = self._clean_llm_response(response_text)
        due_dates = []
        for line in lines:
            due_date = line.strip()
            if due_date.lower() == 'none':
                due_dates.append(None)
            else:
                # Additional filtering for due dates
                if not re.match(r'^(here|the|extracted|timeframes?)', due_date.lower()):
                    due_dates.append(due_date)
                else:
                    due_dates.append(None)
        
        # Fill missing with None
        while len(due_dates) < len(tasks):
            due_dates.append(None)
            
        return due_dates[:len(tasks)]

    def _extract_tags_llm(self, tasks: List[str]) -> List[List[str]]:
        """Extract relevant tags for each task using LLM."""
        tasks_text = '\n'.join(f"{i+1}. {task}" for i, task in enumerate(tasks))
        
        prompt = f"""For each task, return tags or "none"

Tasks:
{tasks_text}

Tags: work, personal, home, shopping, health, finance, social, travel, learning, maintenance

Return tags separated by commas, one line per task:"""

        response = self._llm.invoke([{"role": "user", "content": prompt}])
        response_text = response.content if hasattr(response, 'content') else str(response)
        if not isinstance(response_text, str):
            response_text = str(response_text)
        
        # Clean and parse tags
        lines = self._clean_llm_response(response_text)
        all_tags = []
        valid_tags = {'work', 'personal', 'home', 'shopping', 'health', 'finance', 'social', 'travel', 'learning', 'maintenance'}
        
        for line in lines:
            line = line.strip()
            if line.lower() == 'none':
                all_tags.append([])
            else:
                tags = []
                for tag in line.split(','):
                    tag = tag.strip().lower()
                    if tag in valid_tags:
                        tags.append(tag)
                all_tags.append(tags)
        
        # Fill missing with empty lists
        while len(all_tags) < len(tasks):
            all_tags.append([])
            
        return all_tags[:len(tasks)]

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Ollama."""
        try:
            response = requests.post(self._embedding_url, json={
                "model": "bge-m3:latest",
                "prompt": text
            })
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 1024
    
    def _store_todo_direct(self, todo: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Store todo directly in Qdrant bypassing Mem0 fact extraction."""
        try:
            # Create embedding from todo content
            content = f"Todo: {todo['title']} - {todo['description']} - Priority: {todo['priority']} - Tags: {','.join(todo['tags'])}"
            embedding = self._get_embedding(content)
            
            # Prepare point for Qdrant
            point_id = todo["uuid"]  # Use proper UUID as the point ID
            
            payload = {
                "type": "todo_item_direct",
                "user_id": user_id,
                "todo_id": todo["id"],
                "title": todo["title"],
                "description": todo["description"],
                "priority": todo["priority"],
                "due_date": todo["due_date"],
                "completed": todo["completed"],
                "tags": todo["tags"],
                "created_at": todo["created_at"],
                "content": content  # For searchability
            }
            
            # Store directly in Qdrant
            self._qdrant_client.upsert(
                collection_name="todo_memories",
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            
            logger.info(f"Stored todo {todo['id']} directly in Qdrant")
            return {"success": True, "id": point_id}
            
        except Exception as e:
            logger.error(f"Failed to store todo {todo['id']} directly: {e}")
            return {"success": False, "error": str(e)}

    def _run(self, user_input: str, user_id: str) -> str:
        """Execute the tool to create todo items with enhanced multi-LLM extraction."""
        try:
            logger.info(f"Processing todo creation for user {user_id}: {user_input}")
            
            # Step 1: Extract individual tasks
            tasks = self._extract_tasks_llm(user_input)
            if not tasks:
                return json.dumps({
                    "status": "no_todos_extracted",
                    "message": "No actionable todo items found in your request",
                    "user_input": user_input,
                })

            # Step 2: Extract priorities, due dates, and tags in parallel
            priorities = self._extract_priorities_llm(tasks)
            due_dates = self._extract_due_dates_llm(tasks)
            tags_list = self._extract_tags_llm(tasks)

            # Step 3: Create structured todos
            extracted_todos = []
            for i, task in enumerate(tasks):
                todo_uuid = uuid.uuid4()
                todo_id = f"todo_{todo_uuid.hex[:8]}"  # Keep this for display
                
                extracted_todos.append({
                    "id": todo_id,
                    "uuid": str(todo_uuid),  # Proper UUID for Qdrant point ID
                    "title": task.strip()[:100],
                    "description": task.strip(),
                    "priority": priorities[i] if i < len(priorities) else "medium",
                    "due_date": due_dates[i] if i < len(due_dates) else None,
                    "completed": False,
                    "created_at": datetime.now().isoformat(),
                    "tags": tags_list[i] if i < len(tags_list) else [],
                })

            logger.info(f"Extracted {len(extracted_todos)} todos: {[t['title'] for t in extracted_todos]}")

            # Store each todo as a separate memory entry
            memory_results = []
            
            for i, todo in enumerate(extracted_todos):
                logger.info(f"[{i+1}/{len(extracted_todos)}] Storing todo '{todo['title']}' with ID {todo['id']}")
                
                # Store directly in Qdrant bypassing Mem0
                direct_result = self._store_todo_direct(todo, user_id)
                memory_results.append(direct_result)
                
                time.sleep(0.1)  # Small delay

            # Compute summary statistics first
            priority_counts = {"high": 0, "medium": 0, "low": 0}
            due_date_count = 0
            all_tags = set()
            
            for todo in extracted_todos:
                priority_counts[todo["priority"]] += 1
                if todo["due_date"]:
                    due_date_count += 1
                all_tags.update(todo["tags"])

            # Store summary with delay to ensure individual todos are processed first
            time.sleep(0.1)  # Small delay to let individual todos be processed
            
            summary_messages = [
                {"role": "user", "content": f"BATCH_SUMMARY: Created {len(extracted_todos)} todos from: {user_input}"},
                {"role": "assistant", "content": f"BATCH_COMPLETE: Successfully created batch of {len(extracted_todos)} todos with IDs: {', '.join([t['id'] for t in extracted_todos])}"}
            ]
            summary_metadata = {
                "type": "todo_batch_creation",
                "todo_count": len(extracted_todos),
                "todo_ids": [todo["id"] for todo in extracted_todos],
                "priority_distribution": priority_counts,
                "has_due_dates": due_date_count,
                "unique_tags": list(all_tags),
                "created_at": datetime.now().isoformat(),
                "batch_id": f"batch_{user_id}_{datetime.now().timestamp()}"  # Unique batch ID
            }
            
            summary_result = self.memory.add(
                messages=summary_messages,
                user_id=user_id,
                metadata=summary_metadata,
            )

            # Return structured data
            result_data = {
                "status": "success",
                "action": "todos_created",
                "user_id": user_id,
                "user_input": user_input,
                "created_count": len(extracted_todos),
                "memory_stored": True,
                "todos": extracted_todos,
                "summary": {
                    "total": len(extracted_todos),
                    "priorities": priority_counts,
                    "with_due_dates": due_date_count,
                    "tags": list(all_tags)
                }
            }
            
            return json.dumps(result_data, indent=2)

        except Exception as e:
            logger.error(f"Error in todo creation: {str(e)}")
            return json.dumps({
                "status": "error",
                "action": "todo_creation_failed",
                "user_input": user_input,
                "user_id": user_id,
                "error": str(e),
                "suggestion": "Please try again with a simpler request."
            }, indent=2)


class ListTodosInput(BaseModel):
    """Input for listing and searching todo items."""

    user_id: str = Field(description="Unique identifier for the user")
    filter_completed: Optional[str] = Field(
        default=None, description="Filter by completion status ('true'=completed, 'false'=pending, null/empty=all)"
    )
    filter_priority: Optional[str] = Field(
        default=None, description="Filter by priority level: 'high', 'medium', 'low'"
    )
    search_query: Optional[str] = Field(
        default=None, description="Search for specific keywords in todo titles/descriptions"
    )


class ListTodosTool(BaseTool):
    """Advanced todo listing and search tool with intelligent memory retrieval.
    
    SYSTEM PROMPT: You are an expert todo retrieval assistant. Your job is to:
    
    1. SEARCH memory efficiently for user's todo items
    2. PARSE and ORGANIZE todo data from memory storage  
    3. FILTER todos based on user criteria (priority, completion, keywords)
    4. PRESENT todos in a clear, actionable format
    
    SEARCH CAPABILITIES:
    - Retrieve all todos for a user
    - Filter by completion status (done vs pending)
    - Filter by priority level (high, medium, low)
    - Search by keywords in titles/descriptions
    - Sort by creation date, due date, priority
    
    OUTPUT FORMAT:
    - Clear categorization by status and priority
    - Include all relevant todo metadata (ID, due dates, tags)
    - Show statistics (total count, completed percentage)
    - Provide actionable next steps
    
    ALWAYS provide context about what todos exist and suggest relevant actions.
    """

    name: str = "list_todos"
    description: str = """List, search, and filter todo items from user's memory with advanced options.

    This tool excels at:
    - Retrieving all stored todos for a user
    - Filtering by priority, completion status, or keywords
    - Providing organized views of todo lists
    - Showing relevant metadata and context
    
    Use this tool when users want to:
    - See their current todos ("Show my todos")
    - Find specific items ("Find todos about work") 
    - Check completed items ("What did I finish today?")
    - Review by priority ("Show my urgent tasks")
    """
    args_schema: type = ListTodosInput

    def __init__(self, memory: Memory, **kwargs):
        super().__init__(**kwargs)
        self._memory = memory
        # Add direct Qdrant client for searching direct-stored todos
        self._qdrant_client = QdrantClient(host="localhost", port=6333)

    @property
    def memory(self) -> Memory:
        """Access the memory instance."""
        if not hasattr(self, '_memory') or self._memory is None:
            raise RuntimeError("Memory not properly initialized in ListTodosTool")
        return self._memory

    def _run(
        self,
        user_id: str,
        filter_completed: Optional[str] = None,
        filter_priority: Optional[str] = None,
        search_query: Optional[str] = None,
    ) -> str:
        """List and filter todo items for a user with enhanced search."""
        try:
            # Simple parsing - convert string to boolean if needed, otherwise ignore filter
            completed_filter: Optional[bool] = None
            if filter_completed and filter_completed.lower().strip() not in ['null', 'none', '']:
                filter_completed_lower = filter_completed.lower().strip()
                if filter_completed_lower in ['true', '1', 'yes', 'on']:
                    completed_filter = True
                elif filter_completed_lower in ['false', '0', 'no', 'off']:
                    completed_filter = False
                # Invalid values are ignored - default to plain search
            
            # Plain vector search approach - search direct-stored todos in Qdrant first
            memories = []
            
            logger.info(f"Direct Qdrant search for user {user_id}")
            try:
                scroll_result = self._qdrant_client.scroll(
                    collection_name="todo_memories",
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                            FieldCondition(key="type", match=MatchValue(value="todo_item_direct"))
                        ]
                    ),
                    limit=100,
                    with_payload=True,
                    with_vectors=False
                )
                
                # Convert Qdrant results to memory format
                if scroll_result and len(scroll_result) > 0:
                    points, next_page_offset = scroll_result
                    for point in points:
                        if point and hasattr(point, 'payload') and point.payload:
                            payload = point.payload
                            memories.append({
                                "metadata": payload,
                                "memory": f"Todo: {payload.get('title', '') if payload else ''}",
                                "created_at": payload.get("created_at", "") if payload else ""
                            })
                    logger.info(f"Found {len(points)} direct todos in Qdrant")
                
            except Exception as qdrant_error:
                logger.error(f"Failed to search direct todos in Qdrant: {qdrant_error}")

            # Fallback to Mem0 search only if no direct todos found
            if not memories:
                logger.info(f"No direct todos found, trying Mem0 search for user {user_id}")
                try:
                    if search_query:
                        query = f"todo items tasks {search_query}"
                    else:
                        query = "todo items tasks created completed"
                    
                    memories_result = self.memory.search(
                        query=query, user_id=user_id, limit=100
                    )
                    
                    # Extract actual results from the response dictionary
                    if isinstance(memories_result, dict) and "results" in memories_result:
                        mem0_memories = memories_result["results"]
                    else:
                        mem0_memories = memories_result if memories_result else []
                    
                    # Ensure mem0_memories is a list
                    if isinstance(mem0_memories, list):
                        memories.extend(mem0_memories)
                        logger.info(f"Found {len(mem0_memories)} Mem0 memories")
                        
                except Exception as mem0_error:
                    logger.warning(f"Mem0 search failed (expected for direct-stored todos): {mem0_error}")

            # Ensure memories is a list
            if not isinstance(memories, list):
                memories = []

            if not memories:
                return json.dumps({
                    "status": "no_memories_found",
                    "message": f"No todos found for user {user_id}",
                    "user_id": user_id,
                    "suggestion": "Try creating some todos first!"
                })

            # Extract todos from memory
            all_todos = []
            completed_todo_ids = set()
            
            for memory_item in memories:
                if not isinstance(memory_item, dict):
                    continue
                    
                metadata = memory_item.get("metadata", {})
                memory_content = memory_item.get("memory", "")
                
                if not isinstance(metadata, dict):
                    continue
                
                # Handle individual todo items - support both old and new storage formats
                if metadata.get("type") in ["todo_item", "todo_item_direct"]:
                    todo_id = metadata.get("todo_id")
                    title = metadata.get("title", "")
                    
                    if todo_id and title and title.strip():
                        all_todos.append({
                            "id": todo_id,
                            "title": title.strip(),
                            "description": metadata.get("description", ""),
                            "priority": metadata.get("priority", "medium"),
                            "due_date": metadata.get("due_date"),
                            "completed": metadata.get("completed", False),
                            "tags": metadata.get("tags", []),
                            "created_at": metadata.get("created_at", "")
                        })
                
                # Handle completion records
                elif metadata.get("type") == "todo_completion":
                    completed_id = metadata.get("todo_id")
                    if completed_id:
                        completed_todo_ids.add(completed_id)
                
                # Fallback: extract from memory content if no structured data
                elif not all_todos and "todo" in memory_content.lower():
                    # Simple extraction from content
                    import re
                    patterns = [
                        r"Created.*?todos?:\s*([^.\n!]+)",
                        r"added?\s+[\"']?([^\"'\n.!]+)[\"']?\s+to",
                        r"todo:?\s*([^.\n!]+)"
                    ]
                    
                    for pattern in patterns:
                        matches = re.findall(pattern, memory_content, re.IGNORECASE)
                        for match in matches:
                            title = match.strip()
                            if title and len(title) > 2:
                                all_todos.append({
                                    "id": f"content_{len(all_todos)+1:04d}",
                                    "title": title,
                                    "completed": False,
                                    "created_at": memory_item.get("created_at", "")
                                })

            # Remove duplicates based on title
            seen_titles = set()
            unique_todos = []
            for todo in all_todos:
                title_key = todo["title"].lower().strip()
                if title_key not in seen_titles:
                    seen_titles.add(title_key)
                    unique_todos.append(todo)

            # Mark completed todos
            for todo in unique_todos:
                if todo["id"] in completed_todo_ids:
                    todo["completed"] = True

            # Apply filters
            filtered_todos = []
            try:
                for todo in unique_todos:
                    # Safe boolean comparison with error handling
                    if completed_filter is not None:
                        todo_completed = todo.get("completed", False)
                        if not isinstance(todo_completed, bool):
                            # Try to convert to boolean if it's not already
                            if isinstance(todo_completed, str):
                                todo_completed = todo_completed.lower() in ['true', '1', 'yes', 'on']
                            else:
                                todo_completed = bool(todo_completed)
                        
                        if todo_completed != completed_filter:
                            continue
                    
                    if filter_priority and todo.get("priority", "medium").lower() != filter_priority.lower():
                        continue
                    
                    if search_query and search_query.lower() not in todo["title"].lower():
                        continue
                    
                    filtered_todos.append(todo)
                    
            except Exception as filter_error:
                logger.error(f"Error during todo filtering: {filter_error}")
                return json.dumps({
                    "status": "error",
                    "action": "filtering_failed", 
                    "user_id": user_id,
                    "error": f"Boolean parsing issue in filtering: {str(filter_error)}",
                    "suggestion": "Try your search without completion filters"
                }, indent=2)

            if not filtered_todos:
                return json.dumps({
                    "status": "no_todos_found",
                    "message": f"No todos found matching your criteria for user {user_id}",
                    "user_id": user_id,
                    "search_query": search_query,
                    "filters": {
                        "completed": completed_filter,
                        "priority": filter_priority
                    }
                })

            # Return structured data instead of formatted string
            pending_todos = [t for t in filtered_todos if not t["completed"]]
            completed_todos = [t for t in filtered_todos if t["completed"]]
            
            result_data = {
                "status": "success",
                "user_id": user_id,
                "search_query": search_query,
                "total_todos": len(filtered_todos),
                "pending_count": len(pending_todos),
                "completed_count": len(completed_todos),
                "filters": {
                    "completed": completed_filter,
                    "priority": filter_priority
                },
                "todos": {
                    "pending": pending_todos,
                    "completed": completed_todos
                }
            }
            
            return json.dumps(result_data, indent=2)

        except Exception as e:
            logger.error(f"Error in ListTodosTool._run: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return json.dumps({
                "status": "error", 
                "action": "todo_retrieval_failed",
                "user_id": user_id,
                "search_query": search_query,
                "error": str(e),
                "error_type": type(e).__name__,
                "debug_info": "Error occurred in ListTodosTool._run method"
            }, indent=2)


class CompleteTodoInput(BaseModel):
    """Input for completing a todo item."""

    todo_id: str = Field(description="Todo item ID to mark as complete")
    user_id: str = Field(description="User identifier")


class CompleteTodoTool(BaseTool):
    """Tool for marking todo items as complete."""

    name: str = "complete_todo"
    description: str = "Mark a todo item as completed and update memory."
    args_schema: type = CompleteTodoInput

    def __init__(self, memory: Memory, **kwargs):
        super().__init__(**kwargs)
        self._memory = memory

    @property
    def memory(self) -> Memory:
        """Access the memory instance."""
        if not hasattr(self, '_memory') or self._memory is None:
            raise RuntimeError("Memory not properly initialized in CompleteTodoTool")
        return self._memory

    def _run(self, todo_id: str, user_id: str) -> str:
        """Mark a todo as complete."""
        try:
            # Add completion to memory
            completion_message = f"Completed todo item: {todo_id}"
            
            # Prepare memory add operation
            messages = [
                {
                    "role": "user",
                    "content": f"I completed the task with ID {todo_id}",
                }
            ]
            metadata = {
                "type": "todo_completion",
                "todo_id": todo_id,
                "completed_at": datetime.now().isoformat(),
            }

            # Log the memory add operation
            logger.info(f"MEMORY ADD - User: {user_id}, Messages: {messages}, Metadata: {metadata}")

            memory_result = self.memory.add(
                messages=messages,
                user_id=user_id,
                metadata=metadata,
            )
            
            # Log the response
            logger.info(f"MEMORY ADD RESPONSE: {memory_result}")

            # Return structured data
            result_data = {
                "status": "success",
                "action": "todo_completed", 
                "todo_id": todo_id,
                "user_id": user_id,
                "completed_at": metadata["completed_at"],
                "memory_stored": True
            }
            
            return json.dumps(result_data, indent=2)

        except Exception as e:
            error_data = {
                "status": "error",
                "action": "todo_completion_failed",
                "todo_id": todo_id,
                "user_id": user_id,
                "error": str(e)
            }
            return json.dumps(error_data, indent=2)


def create_todo_tools(memory: Memory) -> List[BaseTool]:
    """Create all todo management tools."""
    return [
        TodoManagerTool(memory=memory),
        ListTodosTool(memory=memory),
        CompleteTodoTool(memory=memory),
    ]


