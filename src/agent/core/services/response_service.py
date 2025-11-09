"""Response service for centralized response formatting and generation."""

from typing import Any, Dict, List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

from src.agent.config import AgentConfig
from src.agent.state import TodoAgentState


class ResponseService:
    """Centralized service for response generation and formatting."""

    def __init__(self, config: AgentConfig):
        self._config = config

    def create_llm(self, temperature: Optional[float] = None) -> Any:
        """Create an LLM instance with the given temperature."""
        return init_chat_model(
            model=self._config.ollama.model,
            model_provider="ollama",
            base_url=self._config.ollama.base_url,
            temperature=temperature or self._config.ollama.temperature,
        )

    def create_todo_system_prompt(self, user_id: str, user_context: str = "") -> str:
        """Create system prompt for todo operations."""
        context_section = ""
        if user_context:
            context_section = f"""

USER CONTEXT (use this to create more relevant todos):
{user_context}

When creating todos, consider the user's context to make them more specific and actionable."""

        return f"""You are Navis, a todo management assistant focused on task operations.

USER ID: {user_id}{context_section}

The user's request has been classified as todo-related. Use the appropriate tools:
- To CREATE todos: call todo_manager with user_input and user_id
- To LIST todos: call list_todos with user_id (and optional filters)
- To COMPLETE todos: call complete_todo with todo_id and user_id

When you receive tool results:
1. Parse the JSON response from the tool
2. Format the data in a user-friendly way with proper organization
3. Use appropriate emojis and formatting for clarity
4. Highlight important information like counts, priorities, due dates
5. Provide helpful context and next steps

Format tool results nicely for the user and provide clear, helpful responses."""

    def create_conversation_system_prompt(self, user_id: str) -> str:
        """Create system prompt for conversation operations."""
        return f"""You are Navis, an intelligent and personable todo management assistant. You excel at having natural conversations while helping users manage their tasks and remembering important details about their lives and work.

USER ID: {user_id}

CONVERSATION STYLE:
- Be warm, friendly, and genuinely interested in the user
- Show that you remember and care about their personal context
- Ask thoughtful follow-up questions to learn more about their goals and preferences
- Provide personalized suggestions based on what you know about them

MEMORY AND PERSONALIZATION:
When users ask personal questions about themselves, you MUST ALWAYS call the search_user_memories tool first. Do not answer from general knowledge.

MANDATORY TOOL USAGE - Call search_user_memories when users ask:
- "What is my name?" → call search_user_memories with query="name"
- "What do I do for work?" → call search_user_memories with query="work job"
- "What are my preferences?" → call search_user_memories with query="preferences"
- "What projects am I working on?" → call search_user_memories with query="projects"
- "Tell me about myself" → call search_user_memories with query="personal information"
- Any question about their personal details, work, hobbies, or past conversations

NEVER answer personal questions without calling the tool first. The user expects you to remember their information.

TODO MANAGEMENT:
For actual todo operations (create, list, complete tasks), guide users to make specific requests. Explain your capabilities when asked, but keep the focus on having a natural conversation and building a relationship.

Be genuinely helpful, show interest in their life, and make them feel heard and understood."""

    def create_final_response_prompt(
        self, user_id: str, response_type: str = "todo"
    ) -> str:
        """Create system prompt for final response generation (no more tool calls)."""
        if response_type == "todo":
            return f"""You are Navis, a todo management assistant. Based on the tool results in the conversation, provide a clear, helpful summary response to the user.

USER ID: {user_id}

DO NOT make any more tool calls. Just summarize what was accomplished based on the tool results shown in the conversation."""
        else:
            return f"""You are Navis, a helpful and intelligent todo management assistant. Based on the memory search results in the conversation, provide a natural, friendly, and personalized response to the user's question.

USER ID: {user_id}

Use the information from the memory search to answer the user's personal question with context and warmth. If the search found relevant information, present it naturally and show that you remember details about them. If no information was found, acknowledge this warmly and encourage them to share more about themselves so you can provide better assistance.

Examples of good responses:
- "Based on what you've told me, your name is Sarah Chen and you work as a Senior Data Scientist at Google..."
- "I remember you mentioning that you work best in the mornings and enjoy rock climbing on weekends..."
- "From our previous conversations, I know you're working on recommendation algorithms at Google..."

DO NOT make any more tool calls. Just provide a conversational response based on the tool results."""

    def build_response_state(
        self,
        messages: List[BaseMessage],
        user_id: str,
        user_context: str = "",
        processing_complete: bool = True,
        todo_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a standard response state dictionary."""
        return {
            "messages": messages,
            "user_id": user_id,
            "user_context": user_context if user_context else None,
            "processing_complete": processing_complete,
            "todo_results": todo_results,
        }

    def create_error_response(
        self, error_msg: str, user_id: str, user_context: str = ""
    ) -> Dict[str, Any]:
        """Create a standardized error response."""
        return self.build_response_state(
            messages=[AIMessage(content=error_msg)],
            user_id=user_id,
            user_context=user_context,
            processing_complete=True,
            todo_results={"error": error_msg},
        )
