from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pydantic import field_validator, BaseModel, Field


class UserInfoType(Enum):
    """Types of user information that can be stored."""
    PERSONAL = "personal"
    PREFERENCE = "preference"
    PROJECT = "project"
    SCHEDULE = "schedule"
    HABIT = "habit"
    
    @classmethod
    def get_all_types(cls) -> List[str]:
        """Get all available info types as strings."""
        return [item.value for item in cls]


class TodoTag(Enum):
    """Available tags for todo items."""
    WORK = "work"
    PERSONAL = "personal"
    HOME = "home"
    SHOPPING = "shopping"
    HEALTH = "health"
    FINANCE = "finance"
    SOCIAL = "social"
    TRAVEL = "travel"
    LEARNING = "learning"
    MAINTENANCE = "maintenance"
    
    @classmethod
    def get_all_tags(cls) -> List[str]:
        """Get all available todo tags as strings."""
        return [item.value for item in cls]


class UserInfoTag(Enum):
    """Available tags for user information."""
    WORK = "work"
    PERSONAL = "personal"
    HEALTH = "health"
    FAMILY = "family"
    HOBBY = "hobby"
    SCHEDULE = "schedule"
    PRIORITY = "priority"
    
    @classmethod
    def get_all_tags(cls) -> List[str]:
        """Get all available user info tags as strings."""
        return [item.value for item in cls]


@dataclass
class TodoItem:
    """A single todo item with structured data."""

    id: str
    title: str
    description: str
    priority: str = "medium"
    due_date: Optional[str] = None
    completed: bool = False
    completed_at: Optional[str] = None
    created_at: str = ""
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "due_date": self.due_date,
            "completed": self.completed,
            "completed_at": self.completed_at,
            "created_at": self.created_at,
            "tags": self.tags,
        }


@dataclass
class UserInfo:
    """User information and context for enhanced todo creation."""
    
    id: str
    user_id: str
    info_type: UserInfoType  # Type of information being stored
    content: str  # The actual information content
    relevance_score: float = 1.0  # How relevant this info is (0.0-1.0)
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    last_used: Optional[str] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "info_type": self.info_type.value if isinstance(self.info_type, UserInfoType) else self.info_type,
            "content": self.content,
            "relevance_score": self.relevance_score,
            "tags": self.tags,
            "created_at": self.created_at,
            "last_used": self.last_used,
        }


@dataclass
class UserContext:
    """Aggregated user context for todo creation."""
    
    user_id: str
    info_by_type: Dict[UserInfoType, List[UserInfo]] = field(default_factory=dict)
    
    def add_info(self, user_info: UserInfo) -> None:
        """Add user information, organizing by type."""
        if user_info.info_type not in self.info_by_type:
            self.info_by_type[user_info.info_type] = []
        self.info_by_type[user_info.info_type].append(user_info)
    
    def get_info_by_type(self, info_type: UserInfoType) -> List[UserInfo]:
        """Get all information of a specific type."""
        return self.info_by_type.get(info_type, [])
    
    def get_context_summary(self, max_per_type: int = 3) -> str:
        """Get a text summary of user context for LLM prompts."""
        context_parts = []
        
        for info_type, infos in self.info_by_type.items():
            if infos:
                # Sort by relevance score and take top items
                sorted_infos = sorted(infos, key=lambda x: x.relevance_score, reverse=True)
                top_infos = sorted_infos[:max_per_type]
                
                content_list = [info.content for info in top_infos]
                if content_list:
                    # Capitalize the first letter of the type name
                    type_name = info_type.value.capitalize()
                    context_parts.append(f"{type_name}: {', '.join(content_list)}")
        
        return " | ".join(context_parts) if context_parts else "No user context available"


class CreateTodosInput(BaseModel):
    """Input for creating multiple todo items."""

    user_input: str = Field(
        description="The user's natural language request to create todo items. Can contain multiple tasks."
    )
    user_id: str = Field(
        description="Unique identifier for the user making the request"
    )

    @field_validator("user_input")
    def validate_user_input(cls, v):
        if not v or not v.strip():
            raise ValueError("User input cannot be empty")
        return v.strip()

    @field_validator("user_id")
    def validate_user_id(cls, v):
        if not v or not v.strip():
            raise ValueError("User ID cannot be empty")
        return v.strip()


class ListTodosInput(BaseModel):
    """Input for listing and searching todo items."""

    user_id: str = Field(description="Unique identifier for the user")
    filter_completed: Optional[str] = Field(
        default=None,
        description="Filter by completion status ('true'=completed, 'false'=pending, null/empty=all)",
    )
    filter_priority: Optional[str] = Field(
        default=None, description="Filter by priority level: 'high', 'medium', 'low'"
    )
    search_query: Optional[str] = Field(
        default=None,
        description="Search for specific keywords in todo titles/descriptions",
    )


class CompleteTodoInput(BaseModel):
    """Input for completing a todo item."""

    todo_id: str = Field(description="Todo item ID to mark as complete")
    user_id: str = Field(description="User identifier")


class BulkCompleteTodoInput(BaseModel):
    """Input for completing multiple todo items at once."""

    todo_ids: List[str] = Field(
        description="List of todo item IDs to mark as complete",
        min_length=1
    )
    user_id: str = Field(description="User identifier")

    @field_validator("todo_ids")
    def validate_todo_ids(cls, v):
        if not v:
            raise ValueError("At least one todo ID must be provided")
        # Remove duplicates while preserving order
        seen = set()
        unique_ids = []
        for todo_id in v:
            if todo_id and todo_id.strip() and todo_id not in seen:
                seen.add(todo_id)
                unique_ids.append(todo_id.strip())
        if not unique_ids:
            raise ValueError("At least one valid todo ID must be provided")
        return unique_ids


class ExtractUserInfoInput(BaseModel):
    """Input for extracting user information from conversation."""
    
    conversation_text: str = Field(description="The conversation text to analyze")
    user_id: str = Field(description="User identifier")


class SearchUserMemoriesInput(BaseModel):
    """Input for searching user memories."""
    
    query: str = Field(description="Search query to find relevant user information")
    user_id: str = Field(description="User identifier")
