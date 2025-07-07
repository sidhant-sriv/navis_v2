"""LangGraph nodes following standard patterns with proper LLM tool calling.

This implementation follows LangGraph best practices from:
https://langchain-ai.github.io/langgraph/tutorials/get-started/5-customize-state/
"""

from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain.chat_models import init_chat_model
from mem0 import Memory
import json
import re

from src.agent.config import AgentConfig
from src.agent.state import TodoAgentState
from src.agent.tools import create_todo_tools


class MemoryManager:
    """Manages Mem0 memory operations with proper error handling."""

    def __init__(self, config: AgentConfig):
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


async def chatbot_node(state: TodoAgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Main chatbot node that processes user messages with LLM and tools.

    This follows the standard LangGraph pattern of using an LLM with bound tools
    to process user messages and decide what actions to take.
    """
    configuration = config.get("configurable", {})
    agent_config = configuration.get("agent_config")
    user_id = configuration.get("user_id", state.get("user_id", "default_user"))

    # Create config from environment if not provided
    if not agent_config:
        agent_config = AgentConfig.from_env()

    # Initialize components
    memory_manager = MemoryManager(agent_config)

    # Create tools with memory
    tools = create_todo_tools(memory_manager.memory, agent_config)

    # Initialize chat model with tools using Ollama configuration
    llm = init_chat_model(
        model=agent_config.ollama.model,
        model_provider="ollama",
        base_url=agent_config.ollama.base_url,
        temperature=agent_config.ollama.temperature,
    )
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
        print(
            "DEBUG: Most recent message is tool result, generating final response without tools"
        )
        # Use LLM without tools to generate final response
        llm = init_chat_model(
            model=agent_config.ollama.model,
            model_provider="ollama",
            base_url=agent_config.ollama.base_url,
            temperature=agent_config.ollama.temperature,
        )

        # Add system message for final response
        final_system_prompt = f"""You are a todo management assistant. Based on the tool results in the conversation, provide a clear, helpful summary response to the user.

USER ID: {user_id}

DO NOT make any more tool calls. Just summarize what was accomplished based on the tool results shown in the conversation."""

        final_messages = [SystemMessage(content=final_system_prompt)] + messages

        try:
            final_response = llm.invoke(final_messages)
            return {
                "messages": [final_response],
                "user_id": user_id,
                "processing_complete": True,
                "todo_results": {"last_action": "final_summary"},
            }
        except Exception as e:
            error_msg = f"Error generating final response: {str(e)}"
            return {
                "messages": [AIMessage(content=error_msg)],
                "user_id": user_id,
                "processing_complete": True,
                "todo_results": {"error": error_msg},
            }

    # Add system message if this is the first interaction
    if not any(msg.type == "system" for msg in messages):
        system_prompt = f"""You are a todo management assistant. You MUST use tools for ALL todo operations.

USER ID: {user_id}

IMPORTANT: Tools return structured JSON data that you must format appropriately for the user.

For todo operations, call the appropriate tool:
- To CREATE todos: call todo_manager with user_input and user_id
- To LIST todos: call list_todos with user_id (and optional filters)
- To COMPLETE todos: call complete_todo with todo_id and user_id

When you receive tool results:
1. Parse the JSON response from the tool
2. Format the data in a user-friendly way with proper organization
3. Use appropriate emojis and formatting for clarity
4. Highlight important information like counts, priorities, due dates
5. Provide helpful context and next steps

Always use proper tool calling format and format the tool results nicely for the user."""

        # Insert system message at the beginning
        messages = [SystemMessage(content=system_prompt)] + messages

    # Invoke LLM with tools
    try:
        response = llm_with_tools.invoke(messages)

        # Check for raw tool call format in the response content
        if hasattr(response, "content") and response.content:
            content = str(response.content)
            print(f"DEBUG: LLM Response content: {content}")
            print(f"DEBUG: Content length: {len(content)}")
            print(f"DEBUG: Content stripped: '{content.strip()}'")
            print(
                f"DEBUG: Starts with {{\"name\": {content.strip().startswith('{\"name\"')}"
            )
            print(f"DEBUG: Contains <|python_tag|>: {'<|python_tag|>' in content}")
            print(f"DEBUG: Contains \"name\": {'\"name\"' in content}")

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
                print("DEBUG: AGGRESSIVE JSON DETECTION TRIGGERED")

                # Extract JSON from anywhere in the content
                try:
                    # Find start and end of JSON
                    start = content.find("{")
                    end = content.rfind("}") + 1
                    if start >= 0 and end > start:
                        extracted_json = content[start:end]
                        print(f"DEBUG: Extracted JSON: {extracted_json}")

                        # Parse and clean
                        tool_data = json.loads(extracted_json)
                        tool_name = tool_data.get("name")
                        tool_params = tool_data.get("parameters", {})

                        # Clean up null values - handle both None and string 'null'
                        cleaned_params = clean_tool_params(tool_params)

                        print(
                            f"DEBUG: AGGRESSIVE - Tool: {tool_name}, Cleaned params: {cleaned_params}"
                        )

                        # Create tool call
                        tool_call = {
                            "name": tool_name,
                            "args": cleaned_params,
                            "id": f"aggressive_{tool_name}",
                        }

                        ai_response = AIMessage(content="", tool_calls=[tool_call])
                        print(f"DEBUG: AGGRESSIVE - Created tool call: {tool_call}")

                        return {
                            "messages": [ai_response],
                            "user_id": user_id,
                            "processing_complete": False,
                            "todo_results": (
                                {"aggressive_parsed": tool_name}
                                if not is_new_human_message
                                else None
                            ),
                        }

                except Exception as e:
                    print(f"DEBUG: AGGRESSIVE parsing failed: {e}")

            # Check if LLM is giving instructions instead of tool calls
            content_lower = content.lower()
            if (
                "call list_todos" in content_lower
                or "to view all todos" in content_lower
            ):
                print("DEBUG: LLM gave list_todos instruction, forcing tool call")
                tool_call = {
                    "name": "list_todos",
                    "args": {"user_id": user_id},
                    "id": "forced_list_todos",
                }
                ai_response = AIMessage(content="", tool_calls=[tool_call])
                return {
                    "messages": [ai_response],
                    "user_id": user_id,
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
                    print("DEBUG: LLM gave todo_manager instruction, forcing tool call")
                    tool_call = {
                        "name": "todo_manager",
                        "args": {"user_input": user_message, "user_id": user_id},
                        "id": "forced_todo_manager",
                    }
                    ai_response = AIMessage(content="", tool_calls=[tool_call])
                    return {
                        "messages": [ai_response],
                        "user_id": user_id,
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
                    print(f"DEBUG: Parsed python_tag tool call (regex): {tool_data}")
                except json.JSONDecodeError as e:
                    print(
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
                            print(
                                f"DEBUG: Parsed python_tag tool call (manual): {tool_data}"
                            )
                        except json.JSONDecodeError as e:
                            print(
                                f"Failed to parse manually extracted python_tag JSON: {complete_json}, error: {e}"
                            )
                            tool_data = None
            elif content.strip().startswith('{"name"'):
                # Try to parse the entire content as JSON
                print(f"DEBUG: Attempting to parse plain JSON: {content.strip()}")
                try:
                    # First, fix common JSON issues like invalid escapes
                    cleaned_content = content.strip()
                    # Fix invalid single quote escapes - replace \' with just '
                    cleaned_content = cleaned_content.replace("\\'", "'")

                    print(f"DEBUG: Cleaned content: {cleaned_content}")
                    tool_data = json.loads(cleaned_content)
                    print(f"DEBUG: Parsed full JSON tool call: {tool_data}")
                except json.JSONDecodeError as e:
                    print(f"Failed to parse full JSON: {content}, error: {e}")
                    # Try one more time with more aggressive cleaning
                    try:
                        # Remove all backslash-single-quote patterns
                        ultra_cleaned = (
                            content.strip().replace("\\'", "'").replace('\\"', '"')
                        )
                        print(f"DEBUG: Ultra-cleaned content: {ultra_cleaned}")
                        tool_data = json.loads(ultra_cleaned)
                        print(
                            f"DEBUG: Parsed ultra-cleaned JSON tool call: {tool_data}"
                        )
                    except json.JSONDecodeError as e2:
                        print(
                            f"Failed to parse ultra-cleaned JSON: {ultra_cleaned}, error: {e2}"
                        )
                        tool_data = None
            else:
                # Fallback: try to find any JSON-like structure in the content
                print(
                    f"DEBUG: No specific pattern matched, trying fallback JSON parsing"
                )
                # Better regex that can handle nested braces
                json_match = re.search(r'\{.*?"name".*?\}', content, re.DOTALL)
                if json_match:
                    try:
                        potential_json = json_match.group(0)
                        print(f"DEBUG: Found potential JSON: {potential_json}")
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
                            print(f"DEBUG: Parsed fallback JSON: {tool_data}")
                        else:
                            tool_data = json.loads(potential_json)
                            print(
                                f"DEBUG: Parsed fallback JSON (no brace counting): {tool_data}"
                            )
                    except json.JSONDecodeError as e:
                        print(
                            f"Failed to parse fallback JSON: {potential_json}, error: {e}"
                        )
                        tool_data = None

                # Last resort: try parsing the entire content if it looks like JSON
                elif "{" in content and "}" in content and '"name"' in content:
                    try:
                        print(
                            f"DEBUG: Last resort - trying to parse entire content as JSON"
                        )
                        # Extract JSON from start of first { to end of last }
                        start = content.find("{")
                        end = content.rfind("}") + 1
                        if start >= 0 and end > start:
                            extracted = content[start:end]
                            tool_data = json.loads(extracted)
                            print(f"DEBUG: Last resort parsing successful: {tool_data}")
                    except json.JSONDecodeError as e:
                        print(f"Last resort parsing failed: {e}")
                        tool_data = None

            if tool_data:
                tool_name = tool_data.get("name")
                tool_params = tool_data.get("parameters", {})

                # Clean up null values - handle both None and string 'null'
                cleaned_params = clean_tool_params(tool_params)

                print(
                    f"DEBUG: Creating tool call - name: {tool_name}, original params: {tool_params}"
                )
                print(f"DEBUG: Cleaned params (nulls removed): {cleaned_params}")

                # Create proper tool call structure
                tool_call = {
                    "name": tool_name,
                    "args": cleaned_params,  # Use cleaned params without nulls
                    "id": f"parsed_{tool_name}",
                }

                # Create AI message with proper tool call
                ai_response = AIMessage(content="", tool_calls=[tool_call])

                print(
                    f"DEBUG: Created AI response with tool calls: {ai_response.tool_calls}"
                )

                return {
                    "messages": [ai_response],
                    "user_id": user_id,
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
            print("DEBUG: Detected completed tool execution, ending conversation")
            return "__end__"

    # If we have any tool messages in recent history and last message is AI without tool calls, end
    has_recent_tool_messages = any(msg.type == "tool" for msg in messages[-5:])
    is_ai_without_tools = isinstance(last_message, AIMessage) and (
        not hasattr(last_message, "tool_calls") or not last_message.tool_calls
    )

    if has_recent_tool_messages and is_ai_without_tools:
        print(
            "DEBUG: Found recent tool execution and AI response without tool calls, ending"
        )
        return "__end__"

    # Otherwise, we're done
    return "__end__"
