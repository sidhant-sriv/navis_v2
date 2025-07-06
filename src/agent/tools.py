"""Tools for todo management with memory integration and improved system prompts."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import re
import logging
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from mem0 import Memory

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
    """Advanced todo creation and management tool with intelligent parsing.
    
    SYSTEM PROMPT: You are an expert todo extraction assistant. Your job is to:
    
    1. PARSE user requests and extract individual actionable todo items
    2. STRUCTURE each todo with proper priority, due dates, and categorization  
    3. STORE todos in memory for future retrieval
    4. PROVIDE clear confirmation of what was created
    
    PARSING RULES:
    - Extract multiple todos from sentences containing 'and', numbered lists, or bullet points
    - Identify priority keywords: "urgent", "asap" = high; "when possible", "eventually" = low
    - Detect due dates: "by Friday", "tomorrow", "next week", etc.
    - Auto-categorize with tags: work, personal, home, shopping, calls, etc.
    
    EXAMPLES:
    ✅ "Add buy groceries and call mom" → 2 separate todos
    ✅ "I need to: 1. Review code 2. Send email" → 2 numbered todos  
    ✅ "Urgent: finish report by Friday" → high priority with due date
    
    ALWAYS confirm what you created and provide todo IDs for reference.
    """

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
        # Use private attribute to avoid Pydantic field validation issues
        self._memory = memory
        self._todo_counter = 1

    @property
    def memory(self) -> Memory:
        """Access the memory instance."""
        if not hasattr(self, '_memory') or self._memory is None:
            raise RuntimeError("Memory not properly initialized in TodoManagerTool")
        return self._memory

    def _extract_todos_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract todo items from natural language text using LLM parsing."""
        todos = []
        logger.info(f"EXTRACTING TODOS from text: '{text}'")

        # Simple approach: split on common separators and clean up
        # The LLM calling this tool has already understood the intent
        
        # Strategy 1: Look for common separators
        potential_items = []
        
        # Split on commas and "and" 
        items = re.split(r',\s*(?:and\s+)?|\s+and\s+', text)
        
        for item in items:
            # Clean up each item
            cleaned = re.sub(
                r'(?:^|\s+)(?:remind me to|i (?:also )?need to|i (?:also )?have to|i (?:also )?should|i (?:also )?must)\s+',
                '', item, flags=re.IGNORECASE
            ).strip()
            
            # Remove leading/trailing punctuation but keep internal punctuation
            cleaned = re.sub(r'^[,\.\s]+|[,\.\s]+$', '', cleaned).strip()
            
            if len(cleaned) > 2:  # Only keep meaningful items
                potential_items.append(cleaned)

        # If no good splits found, try to extract the main action from the whole text
        if not potential_items:
            # Remove common prefixes and use the rest
            cleaned = re.sub(
                r'(?:^|\s+)(?:add|create|make|todo|task|remind me to|i need to|i have to|i should|i must)\s+',
                '', text, flags=re.IGNORECASE
            ).strip()
            if len(cleaned) > 2:
                potential_items.append(cleaned)

        logger.info(f"Extracted items: {potential_items}")

        # Convert to structured todos
        for item in potential_items:
            if not item.strip():
                continue
                
            # Simple priority detection
            priority = "medium"
            if any(word in item.lower() for word in ["urgent", "asap", "critical", "emergency"]):
                priority = "high"
            elif any(word in item.lower() for word in ["eventually", "sometime", "later"]):
                priority = "low"

            # Simple due date extraction
            due_date = None
            if re.search(r'by\s+\w+|today|tomorrow|this week|next week', item.lower()):
                due_match = re.search(r'(by\s+\w+|today|tomorrow|this week|next week)', item.lower())
                if due_match:
                    due_date = due_match.group(1)

            # Simple tag detection
            tags = []
            tag_keywords = {
                "work": ["work", "project", "meeting", "report", "email"],
                "personal": ["mom", "family", "friend", "call"],
                "home": ["clean", "fix", "house"],
                "shopping": ["buy", "grocery", "store"],
            }
            
            item_lower = item.lower()
            for tag, keywords in tag_keywords.items():
                if any(keyword in item_lower for keyword in keywords):
                    tags.append(tag)

            # Generate unique ID
            todo_id = f"todo_{self._todo_counter:04d}"
            self._todo_counter += 1

            todos.append({
                "id": todo_id,
                "title": item.strip()[:100],
                "description": item.strip(),
                "priority": priority,
                "due_date": due_date,
                "completed": False,
                "created_at": datetime.now().isoformat(),
                "tags": tags,
            })

        logger.info(f"Final todos: {[t['title'] for t in todos]}")
        return todos

    def _run(self, user_input: str, user_id: str) -> str:
        """Execute the tool to create todo items with enhanced feedback."""
        try:
            # Extract todos from the input
            extracted_todos = self._extract_todos_from_text(user_input)

            if not extracted_todos:
                return json.dumps({
                    "status": "no_todos_extracted",
                    "message": "No actionable todo items found in your request",
                    "user_input": user_input,
                    "suggestions": [
                        "Add buy groceries to my todos",
                        "Create task: finish report by Friday",
                        "I need to: 1. Call doctor 2. Pay bills",
                        "Remind me to walk the dog and feed the cat"
                    ]
                })

            # Store in memory with structured data
            memory_content = f"User created {len(extracted_todos)} todo items"
            todo_titles = [todo["title"] for todo in extracted_todos]
            
            # Prepare memory add operation
            messages = [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": f"Created {len(extracted_todos)} todos: {', '.join(todo_titles)}"}
            ]
            metadata = {
                "type": "todo_creation",
                "todo_count": len(extracted_todos),
                "todo_ids": [todo["id"] for todo in extracted_todos],
                "todo_titles": todo_titles,
                "priority_distribution": {
                    "high": len([t for t in extracted_todos if t["priority"] == "high"]),
                    "medium": len([t for t in extracted_todos if t["priority"] == "medium"]), 
                    "low": len([t for t in extracted_todos if t["priority"] == "low"])
                },
                "has_due_dates": len([t for t in extracted_todos if t["due_date"]]),
                "tags": list(set([tag for todo in extracted_todos for tag in todo["tags"]])),
                "created_at": datetime.now().isoformat()
            }
            
            # Log the memory add operation
            logger.info(f"MEMORY ADD - User: {user_id}, Messages: {messages}, Metadata: {metadata}")
            
            # Store with rich metadata for better retrieval
            memory_result = self.memory.add(
                messages=messages,
                user_id=user_id,
                metadata=metadata,
            )
            
            # Log the response
            logger.info(f"MEMORY ADD RESPONSE: {memory_result}")

            # Return structured data instead of formatted response
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
                    "priorities": metadata["priority_distribution"],
                    "with_due_dates": metadata["has_due_dates"],
                    "tags": metadata["tags"]
                }
            }
            
            return json.dumps(result_data, indent=2)

        except Exception as e:
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
    filter_completed: Optional[bool] = Field(
        default=None, description="Filter by completion status (true=completed, false=pending, null=all)"
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

    @property
    def memory(self) -> Memory:
        """Access the memory instance."""
        if not hasattr(self, '_memory') or self._memory is None:
            raise RuntimeError("Memory not properly initialized in ListTodosTool")
        return self._memory

    def _run(
        self,
        user_id: str,
        filter_completed: Optional[bool] = None,
        filter_priority: Optional[str] = None,
        search_query: Optional[str] = None,
    ) -> str:
        """List and filter todo items for a user with enhanced search."""
        try:
            # Search memory for todo-related content
            if search_query:
                query = f"todo items tasks {search_query}"
            else:
                query = "todo items tasks created completed"
            
            # Log the memory search operation
            logger.info(f"MEMORY SEARCH - User: {user_id}, Query: '{query}', Limit: 100")
            
            memories_result = self.memory.search(
                query=query, user_id=user_id, limit=100
            )
            
            # Log the response
            logger.info(f"MEMORY SEARCH RESPONSE: {memories_result}")

            # Extract actual results from the response dictionary
            if isinstance(memories_result, dict) and "results" in memories_result:
                memories = memories_result["results"]
            else:
                memories = memories_result if memories_result else []

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
                
                # Handle todo creation records
                if metadata.get("type") == "todo_creation":
                    todo_ids = metadata.get("todo_ids", [])
                    todo_titles = metadata.get("todo_titles", [])
                    created_at = metadata.get("created_at", "")
                    
                    # Extract todos from metadata
                    for todo_id, title in zip(todo_ids, todo_titles):
                        if title and title.strip() and len(title.strip()) > 1:  # Valid title
                            all_todos.append({
                                "id": todo_id,
                                "title": title.strip(),
                                "completed": False,
                                "created_at": created_at
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
            for todo in unique_todos:
                if filter_completed is not None and todo["completed"] != filter_completed:
                    continue
                if search_query and search_query.lower() not in todo["title"].lower():
                    continue
                filtered_todos.append(todo)

            if not filtered_todos:
                return json.dumps({
                    "status": "no_todos_found",
                    "message": f"No todos found matching your criteria for user {user_id}",
                    "user_id": user_id,
                    "search_query": search_query,
                    "filters": {
                        "completed": filter_completed,
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
                    "completed": filter_completed,
                    "priority": filter_priority
                },
                "todos": {
                    "pending": pending_todos,
                    "completed": completed_todos
                }
            }
            
            return json.dumps(result_data, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error", 
                "action": "todo_retrieval_failed",
                "user_id": user_id,
                "search_query": search_query,
                "error": str(e)
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


