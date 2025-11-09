"""Core nodes for the Todo Agent, including intent routing, todo chatbot, and conversation chatbot."""

import json
import logging
import re
from typing import Any, Dict

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from mem0 import Memory

from src.agent.config import AgentConfig
from src.agent.core.services import ServiceContainer
from src.agent.state import TodoAgentState


# Legacy MemoryManager for backward compatibility during migration
class MemoryManager:
    """Manages Mem0 memory operations with proper error handling."""

    def __init__(self, config: AgentConfig):
        """Initialize MemoryManager with given configuration."""
        self.config = config
        self._memory = None

    @property
    def memory(self) -> Memory:
        """Lazy initialization of memory."""
        if self._memory is None:
            mem0_config = self.config.get_mem0_config()
            self._memory = Memory.from_config(mem0_config)
        return self._memory


def clean_tool_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Clean up null values and invalid parameters from tool parameters."""
    cleaned_params = {}
    for key, value in params.items():
        # Skip various forms of null/empty values
        if (
            value is not None
            and value != "null"
            and value != ""
            and str(value).lower() != "null"
        ):
            # Special handling for boolean fields - enhanced to handle more formats
            if key in ["filter_completed"] and isinstance(value, str):
                value_lower = value.lower().strip()
                if value_lower in ["true", "1", "yes", "on"]:
                    cleaned_params[key] = True
                elif value_lower in ["false", "0", "no", "off"]:
                    cleaned_params[key] = False
                # Skip invalid boolean strings (don't add to cleaned_params)
            elif key in ["filter_completed"] and isinstance(value, bool):
                # Already a boolean, use as-is
                cleaned_params[key] = value
            else:
                cleaned_params[key] = value
    return cleaned_params


async def todo_chatbot_node(
    state: TodoAgentState, config: RunnableConfig
) -> Dict[str, Any]:
    """Todo-focused chatbot node that processes todo operations with LLM and tools.

    This node is only called when intent routing determines the user wants todo operations.
    It follows the standard LangGraph pattern of using an LLM with bound tools.
    """
    # Initialize services using the new service container
    services = ServiceContainer.from_config(config)
    user_id = services.get_user_id(config, dict(state))

    # Retrieve user context for enhanced todo creation
    user_context_summary = services.memory.get_user_context_summary(user_id)
    if user_context_summary:
        (f"DEBUG: Retrieved user context: {user_context_summary}")

    # Get tools from service
    tools = services.tools.get_tools_for_context("todo")

    # Initialize chat model with tools using Ollama configuration
    llm = services.responses.create_llm()
    llm_with_tools = llm.bind_tools(tools)

    # Get messages from state
    messages = state.get("messages", [])

    # Check if we just received tool results - only disable tools if the most recent message is a tool result
    # If the most recent message is human, we should allow tools regardless of history
    most_recent_message = messages[-1] if messages else None
    has_immediate_tool_results = (
        most_recent_message and most_recent_message.type == "tool"
    )

    # Reset processing state for new human messages
    is_new_human_message = most_recent_message and most_recent_message.type == "human"

    if has_immediate_tool_results:
        logging.debug(
            "DEBUG: Most recent message is tool result, generating final response without tools"
        )
        # Use LLM without tools to generate final response
        llm = services.responses.create_llm()

        # Add system message for final response
        final_system_prompt = services.responses.create_final_response_prompt(
            user_id, "todo"
        )

        final_messages = [SystemMessage(content=final_system_prompt)] + messages

        try:
            final_response = llm.invoke(final_messages)
            return services.responses.build_response_state(
                messages=[final_response],
                user_id=user_id,
                user_context=user_context_summary,
                processing_complete=True,
                todo_results={"last_action": "final_summary"},
            )
        except Exception as e:
            error_msg = f"Error generating final response: {str(e)}"
            return services.responses.create_error_response(
                error_msg, user_id, user_context_summary
            )

    # Add system message if this is the first interaction
    if not any(msg.type == "system" for msg in messages):
        system_prompt = services.responses.create_todo_system_prompt(
            user_id, user_context_summary
        )
        # Insert system message at the beginning
        messages = [SystemMessage(content=system_prompt)] + messages

    # Invoke LLM with tools
    try:
        response = llm_with_tools.invoke(messages)

        # Check for raw tool call format in the response content
        if hasattr(response, "content") and response.content:
            content = str(response.content)
            logging.debug(f"LLM Response content: {content}")
            logging.debug(f"Content length: {len(content)}")
            logging.debug(f"Content stripped: '{content.strip()}'")
            logging.debug(
                f'Starts with {{"name": {content.strip().startswith('{"name"')}'
            )
            logging.debug(f"Contains <|python_tag|>: {'<|python_tag|>' in content}")
            logging.debug(f'Contains "name": {'"name"' in content}')

            # AGGRESSIVE JSON DETECTION - trigger if content looks like JSON with "name" field
            if (
                "{" in content
                and "}" in content
                and '"name"' in content
                and (
                    "list_todos" in content
                    or "todo_manager" in content
                    or "complete_todo" in content
                )
            ):
                # Extract JSON from anywhere in the content
                try:
                    # Find start and end of JSON
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    if start >= 0 and end > start:
                        extracted_json = content[start:end]
                        (f"DEBUG: Extracted JSON: {extracted_json}")

                        # Parse and clean
                        tool_data = json.loads(extracted_json)
                        tool_name = tool_data.get("name")
                        tool_params = tool_data.get("parameters", {})

                        # Clean up null values - handle both None and string 'null'
                        cleaned_params = clean_tool_params(tool_params)

                        logging.debug(
                            f"DEBUG: AGGRESSIVE - Tool: {tool_name}, Cleaned params: {cleaned_params}"
                        )

                        # Create tool call
                        tool_call = {
                            "name": tool_name,
                            "args": cleaned_params,
                            "id": f"aggressive_{tool_name}",
                        }

                        ai_response = AIMessage(content="", tool_calls=[tool_call])
                        logging.debug(
                            f"DEBUG: AGGRESSIVE - Created tool call: {tool_call}"
                        )

                        return {
                            "messages": [ai_response],
                            "user_id": user_id,
                            "user_context": (
                                user_context_summary if user_context_summary else None
                            ),
                            "processing_complete": False,
                            "todo_results": (
                                {"aggressive_parsed": tool_name}
                                if not is_new_human_message
                                else None
                            ),
                        }

                except Exception as e:
                    logging.debug(f"DEBUG: AGGRESSIVE parsing failed: {e}")

            # Check if LLM is giving instructions instead of tool calls
            content_lower = content.lower()
            if (
                "call list_todos" in content_lower
                or "to view all todos" in content_lower
            ):
                ("DEBUG: LLM gave list_todos instruction, forcing tool call")
                tool_call = {
                    "name": "list_todos",
                    "args": {"user_id": user_id},
                    "id": "forced_list_todos",
                }
                ai_response = AIMessage(content="", tool_calls=[tool_call])
                return {
                    "messages": [ai_response],
                    "user_id": user_id,
                    "user_context": (
                        user_context_summary if user_context_summary else None
                    ),
                    "processing_complete": False,
                    "todo_results": (
                        {"forced_tool": "list_todos"}
                        if not is_new_human_message
                        else None
                    ),
                }
            elif (
                "call todo_manager" in content_lower
                or "to create todos" in content_lower
            ):
                # Get the original user message to pass to todo_manager
                user_message = None
                for msg in reversed(messages):
                    if msg.type == "human":
                        user_message = msg.content
                        break

                if user_message:
                    ("DEBUG: LLM gave todo_manager instruction, forcing tool call")
                    tool_call = {
                        "name": "todo_manager",
                        "args": {"user_input": user_message, "user_id": user_id},
                        "id": "forced_todo_manager",
                    }
                    ai_response = AIMessage(content="", tool_calls=[tool_call])
                    return {
                        "messages": [ai_response],
                        "user_id": user_id,
                        "user_context": (
                            user_context_summary if user_context_summary else None
                        ),
                        "processing_complete": False,
                        "todo_results": (
                            {"forced_tool": "todo_manager"}
                            if not is_new_human_message
                            else None
                        ),
                    }

            # Parse raw tool call format like <|python_tag|>{"name": "list_todos", "parameters": {"user_id":"default"}}
            tool_call_pattern = r"<\|python_tag\|>(\{.*?\}(?:\})?)"
            match = re.search(tool_call_pattern, content)

            # Better approach - find the JSON after python_tag
            tool_data = None
            if match:
                try:
                    tool_data = json.loads(match.group(1))
                    (f"DEBUG: Parsed python_tag tool call (regex): {tool_data}")
                except json.JSONDecodeError as e:
                    (
                        f"Failed to parse python_tag tool call (regex): {match.group(1)}, error: {e}"
                    )
            elif "<|python_tag|>" in content:
                start_idx = content.find("<|python_tag|>") + len("<|python_tag|>")
                json_part = content[start_idx:].strip()
                if json_part.startswith("{"):
                    # Find the matching closing brace
                    brace_count = 0
                    end_idx = 0
                    for i, char in enumerate(json_part):
                        if char == "{":
                            brace_count += 1
                        elif char == "}":
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i + 1
                                break

                    if end_idx > 0:
                        complete_json = json_part[:end_idx]
                        try:
                            tool_data = json.loads(complete_json)
                            (
                                f"DEBUG: Parsed python_tag tool call (manual): {tool_data}"
                            )
                        except json.JSONDecodeError as e:
                            (
                                f"Failed to parse manually extracted python_tag JSON: {complete_json}, error: {e}"
                            )
                            tool_data = None
            elif content.strip().startswith('{"name"'):
                # Try to parse the entire content as JSON
                (f"DEBUG: Attempting to parse plain JSON: {content.strip()}")
                try:
                    # First, fix common JSON issues like invalid escapes
                    cleaned_content = content.strip()
                    # Fix invalid single quote escapes - replace \' with just '
                    cleaned_content = cleaned_content.replace("\\'", "'")

                    (f"DEBUG: Cleaned content: {cleaned_content}")
                    tool_data = json.loads(cleaned_content)
                    (f"DEBUG: Parsed full JSON tool call: {tool_data}")
                except json.JSONDecodeError as e:
                    (f"Failed to parse full JSON: {content}, error: {e}")
                    # Try one more time with more aggressive cleaning
                    try:
                        # Remove all backslash-single-quote patterns
                        ultra_cleaned = (
                            content.strip().replace("\\'", "'").replace('\\"', '"')
                        )
                        logging.debug("DEBUG: Ultra-cleaned content")
                        tool_data = json.loads(ultra_cleaned)
                        logging.debug(
                            f"DEBUG: Parsed ultra-cleaned JSON tool call: {tool_data}"
                        )
                    except json.JSONDecodeError as e2:
                        logging.debug(
                            f"Failed to parse ultra-cleaned JSON, error: {e2}",
                        )
                        tool_data = None
            else:
                # Fallback: try to find any JSON-like structure in the content
                logging.debug(
                    "DEBUG: No specific pattern matched, trying fallback JSON parsing"
                )
                # Better regex that can handle nested braces
                json_match = re.search(r'\{.*?"name".*?\}', content, re.DOTALL)
                if json_match:
                    try:
                        potential_json = json_match.group(0)
                        (f"DEBUG: Found potential JSON: {potential_json}")
                        # Count braces to find complete JSON
                        brace_count = 0
                        end_pos = 0
                        for i, char in enumerate(potential_json):
                            if char == "{":
                                brace_count += 1
                            elif char == "}":
                                brace_count -= 1
                                if brace_count == 0:
                                    end_pos = i + 1
                                    break

                        if end_pos > 0:
                            complete_json = potential_json[:end_pos]
                            tool_data = json.loads(complete_json)
                            (f"DEBUG: Parsed fallback JSON: {tool_data}")
                        else:
                            tool_data = json.loads(potential_json)
                            (
                                f"DEBUG: Parsed fallback JSON (no brace counting): {tool_data}"
                            )
                    except json.JSONDecodeError as e:
                        (f"Failed to parse fallback JSON: {potential_json}, error: {e}")
                        tool_data = None

                # Last resort: try parsing the entire content if it looks like JSON
                elif "{" in content and "}" in content and '"name"' in content:
                    try:
                        (f"DEBUG: Last resort - trying to parse entire content as JSON")
                        # Extract JSON from start of first { to end of last }
                        start = content.find("{")
                        end = content.rfind("}") + 1
                        if start >= 0 and end > start:
                            extracted = content[start:end]
                            tool_data = json.loads(extracted)
                            (f"DEBUG: Last resort parsing successful: {tool_data}")
                    except json.JSONDecodeError as e:
                        (f"Last resort parsing failed: {e}")
                        tool_data = None

            if tool_data:
                tool_name = tool_data.get("name")
                tool_params = tool_data.get("parameters", {})

                # Clean up null values - handle both None and string 'null'
                cleaned_params = clean_tool_params(tool_params)

                (
                    f"DEBUG: Creating tool call - name: {tool_name}, original params: {tool_params}"
                )
                (f"DEBUG: Cleaned params (nulls removed): {cleaned_params}")

                # Create proper tool call structure
                tool_call = {
                    "name": tool_name,
                    "args": cleaned_params,  # Use cleaned params without nulls
                    "id": f"parsed_{tool_name}",
                }

                # Create AI message with proper tool call
                ai_response = AIMessage(content="", tool_calls=[tool_call])

                (
                    f"DEBUG: Created AI response with tool calls: {ai_response.tool_calls}"
                )

                return {
                    "messages": [ai_response],
                    "user_id": user_id,
                    "user_context": (
                        user_context_summary if user_context_summary else None
                    ),
                    "processing_complete": False,
                    "todo_results": (
                        {"parsed_tool": tool_name} if not is_new_human_message else None
                    ),
                }

        # Check if response has tool calls (only AIMessage has this attribute)
        has_tool_calls = (
            isinstance(response, AIMessage)
            and hasattr(response, "tool_calls")
            and bool(response.tool_calls)
        )

        # Update state with user_id and processing status
        return {
            "messages": [response],
            "user_id": user_id,
            "user_context": user_context_summary if user_context_summary else None,
            "processing_complete": not has_tool_calls,
            "todo_results": (
                {"last_action": "chatbot_processing"}
                if not is_new_human_message
                else None
            ),
        }

    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        return {
            "messages": [AIMessage(content=error_msg)],
            "user_id": user_id,
            "user_context": user_context_summary if user_context_summary else None,
            "processing_complete": True,
            "todo_results": {"error": error_msg},
        }


def should_continue(state: TodoAgentState) -> str:
    """Determine if we should continue to tools or end."""
    messages = state.get("messages", [])
    if not messages:
        return "__end__"

    last_message = messages[-1]

    # Check if the last message has tool calls (only AIMessage has this attribute)
    if (
        isinstance(last_message, AIMessage)
        and hasattr(last_message, "tool_calls")
        and last_message.tool_calls
    ):
        return "tools"

    # Check if we just came back from tool execution
    # Look for pattern: tool_call -> tool_result -> ai_response (without tool calls)
    if len(messages) >= 3:
        # Check if the last few messages show we just completed tool execution
        recent_msg_types = [msg.type for msg in messages[-3:]]
        if (
            "tool" in recent_msg_types
            and isinstance(last_message, AIMessage)
            and (not hasattr(last_message, "tool_calls") or not last_message.tool_calls)
        ):
            ("DEBUG: Detected completed tool execution, ending conversation")
            return "__end__"

    # If we have any tool messages in recent history and last message is AI without tool calls, end
    has_recent_tool_messages = any(msg.type == "tool" for msg in messages[-5:])
    is_ai_without_tools = isinstance(last_message, AIMessage) and (
        not hasattr(last_message, "tool_calls") or not last_message.tool_calls
    )

    if has_recent_tool_messages and is_ai_without_tools:
        (
            "DEBUG: Found recent tool execution and AI response without tool calls, ending"
        )
        return "__end__"

    # Otherwise, we're done
    return "__end__"


def get_latest_human_message(messages: list) -> str:
    """Extract the latest human message from the message list."""
    for message in reversed(messages):
        if hasattr(message, "type") and message.type == "human":
            content = message.content
            # Handle list content (e.g., multimodal messages)
            if isinstance(content, list):
                # Extract text from list of content parts
                text_parts = [
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                ]
                return " ".join(text_parts).strip()
            return str(content) if content else ""
    return ""


def fallback_pattern_match(user_input: str) -> str:
    """Simple pattern matching fallback for intent classification."""
    # Handle case where user_input might be a list
    if isinstance(user_input, list):
        text_parts = [
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in user_input
        ]
        text = " ".join(text_parts).strip().lower()
    else:
        text = str(user_input).lower() if user_input else ""

    # Strong todo indicators
    todo_patterns = [
        "add",
        "create",
        "make",
        "remind",
        "schedule",
        "list",
        "show",
        "display",
        "what are my",
        "complete",
        "done",
        "finish",
        "mark",
        "todo",
        "task",
        "reminder",
        "need to",
        "have to",
    ]

    # Strong conversation indicators
    conversation_patterns = [
        "hello",
        "hi",
        "hey",
        "how are you",
        "what can you",
        "help",
        "explain",
        "tell me",
        "weather",
        "how do",
    ]

    if any(pattern in text for pattern in todo_patterns):
        return "todo"
    elif any(pattern in text for pattern in conversation_patterns):
        return "conversation"
    else:
        # Default to conversation for ambiguous cases
        return "conversation"


async def intent_router_node(
    state: TodoAgentState, config: RunnableConfig
) -> Dict[str, Any]:
    """Simple binary router: todo vs conversation."""

    messages = state.get("messages", [])
    user_message = get_latest_human_message(messages)

    if not user_message:
        # No user message found, default to conversation
        return {"detected_intent": "conversation", "processing_complete": False}

    # Simple classification prompt - just binary choice
    classification_prompt = f"""
    Determine if this user input requires todo operations or is general conversation.
    
    Input: "{user_message}"
    
    Todo operations include:
    - Creating tasks ("add", "create", "remind me to")
    - Listing tasks ("show", "list", "what are my")  
    - Completing tasks ("mark done", "complete", "finished")
    - Task-related questions ("how many tasks", "what's my next")
    
    General conversation includes:
    - Greetings ("hello", "hi", "how are you")
    - Questions about the system ("what can you do", "help")
    - General questions ("weather", "explain", "tell me about")
    
    Reply with only: "todo" or "conversation"
    """

    # Get LLM classification
    services = ServiceContainer.from_config(config)
    llm = services.responses.create_llm(temperature=0.0)  # Deterministic responses

    try:
        response = llm.invoke([{"role": "user", "content": classification_prompt}])
        content = response.content if hasattr(response, "content") else str(response)
        intent = str(content).strip().lower()

        # Fallback pattern matching if LLM gives unexpected response
        if intent not in ["todo", "conversation"]:
            intent = fallback_pattern_match(user_message)

        (
            f"DEBUG: Intent classification - Input: '{user_message}' -> Intent: '{intent}'"
        )

    except Exception as e:
        (f"DEBUG: Intent classification failed: {e}, using fallback")
        intent = fallback_pattern_match(user_message)

    return {"detected_intent": intent, "processing_complete": False}


async def conversation_chatbot_node(
    state: TodoAgentState, config: RunnableConfig
) -> Dict[str, Any]:
    """Handle general conversation with memory search capabilities."""

    # Initialize services using the new service container
    services = ServiceContainer.from_config(config)
    user_id = services.get_user_id(config, dict(state))

    # Get the latest user message for information extraction
    messages = state.get("messages", [])
    user_message = get_latest_human_message(messages)

    # Check if this is a fresh human message (not a tool result processing)
    most_recent_message = messages[-1] if messages else None
    is_fresh_human_message = (
        most_recent_message
        and most_recent_message.type == "human"
        and not any(
            msg.type == "tool" for msg in messages[-3:]
        )  # No recent tool messages
    )

    # Only extract user information on fresh human messages to avoid duplicates
    if is_fresh_human_message and services.memory.should_extract_info_from_message(
        user_message, bool(state.get("user_info_extracted", False))
    ):
        extracted_infos = services.memory.extract_and_store_user_info(
            user_message, user_id
        )
        if extracted_infos:
            (f"DEBUG: Extracted {len(extracted_infos)} pieces of user information")
            # Mark that we've extracted info for this conversation turn
            state["user_info_extracted"] = True

    # Get tools from service
    conversation_tools = services.tools.get_tools_for_context("conversation")

    # Initialize LLM with tools for memory search capabilities
    llm = services.responses.create_llm(temperature=0.3)
    llm_with_tools = llm.bind_tools(conversation_tools)

    # Check if we just received tool results
    most_recent_message = messages[-1] if messages else None
    has_immediate_tool_results = (
        most_recent_message and most_recent_message.type == "tool"
    )

    if has_immediate_tool_results:
        ("DEBUG: Conversation node processing tool results")
        # Use LLM without tools to generate final response based on tool results
        llm = services.responses.create_llm(temperature=0.3)

        final_system_prompt = services.responses.create_final_response_prompt(
            user_id, "conversation"
        )

        final_messages = [SystemMessage(content=final_system_prompt)] + messages

        try:
            response = llm.invoke(final_messages)
            return services.responses.build_response_state(
                messages=[response],
                user_id=user_id,
                processing_complete=True,
                todo_results={"interaction_type": "conversation_with_memory"},
            )
        except Exception as e:
            error_msg = f"Error generating memory-based response: {str(e)}"
            return services.responses.create_error_response(error_msg, user_id)

    # Add system message for conversation with memory search capabilities
    if not any(msg.type == "system" for msg in messages):
        system_prompt = services.responses.create_conversation_system_prompt(user_id)
        messages = [SystemMessage(content=system_prompt)] + messages

    try:
        response = llm_with_tools.invoke(messages)

        return services.responses.build_response_state(
            messages=[response],
            user_id=user_id,
            processing_complete=False,  # May need to handle tool calls
            todo_results={"interaction_type": "conversation"},
        )
    except Exception as e:
        error_msg = f"Error in conversation: {str(e)}"
        return services.responses.create_error_response(error_msg, user_id)


def simple_route_decision(state: TodoAgentState) -> str:
    """Simple routing without confidence thresholds."""
    intent = state.get("detected_intent", "conversation")

    (
        f"DEBUG: Routing decision - Intent: '{intent}' -> Route: {'todo_chatbot' if intent == 'todo' else 'conversation_chatbot'}"
    )

    if intent == "todo":
        return "todo_chatbot"
    else:
        return "conversation_chatbot"
