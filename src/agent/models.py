from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import field_validator, BaseModel, Field


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
