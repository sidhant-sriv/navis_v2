"""LangGraph todo management agent following standard patterns.

A complete todo management system using LangGraph, Mem0, Qdrant, and Ollama
following best practices from: https://langchain-ai.github.io/langgraph/tutorials/
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage

from src.agent.config import AgentConfig
from src.agent.state import TodoAgentState
from src.agent.nodes import chatbot_node, should_continue, MemoryManager
from src.agent.tools import create_todo_tools


class Configuration(TypedDict):
    """Configurable parameters for the todo agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    agent_config: Optional[AgentConfig]
    user_id: str


def create_todo_graph() -> StateGraph:
    """Create the todo management graph following standard LangGraph patterns."""

    # Create the graph with TodoAgentState and Configuration
    graph = StateGraph(TodoAgentState, config_schema=Configuration)

    # We'll bind tools in the chatbot node, but we need to create them here for the ToolNode
    # The actual tools will be created with memory in the chatbot node
    # For now, create a placeholder - the real tools are created in chatbot_node

    # Add the main chatbot node
    graph.add_node("chatbot", chatbot_node)

    # Add tools node - we'll use a simple approach and create tools with default config
    # The tools will be re-created in each node call but that's acceptable for this use case
    def tools_node(state: TodoAgentState, config: RunnableConfig):
        """Tools node that creates tools with proper memory configuration."""
        configuration = config.get("configurable", {})
        agent_config = configuration.get("agent_config")

        if not agent_config:
            agent_config = AgentConfig.from_env()

        memory_manager = MemoryManager(agent_config)
        tools = create_todo_tools(memory_manager.memory)

        # Use ToolNode to execute the tools
        tool_executor = ToolNode(tools)
        return tool_executor.invoke(state, config)

    graph.add_node("tools", tools_node)

    # Set up the flow: START -> chatbot
    graph.add_edge(START, "chatbot")

    # Add conditional edges from chatbot
    graph.add_conditional_edges(
        "chatbot",
        should_continue,
        {
            "tools": "tools",
            "__end__": END,
        },
    )

    # After tools, go back to chatbot for final response, but should_continue will determine if we end
    graph.add_edge("tools", "chatbot")

    return graph


# Create the compiled graph
graph = create_todo_graph().compile(name="Todo Management Agent")


# Helper function to run the graph with proper configuration
async def run_todo_agent(
    user_input: str, user_id: str, agent_config: Optional[AgentConfig] = None
) -> Dict[str, Any]:
    """Run the todo agent with the given input.

    Args:
        user_input: The user's todo-related request
        user_id: Unique identifier for the user
        agent_config: Configuration for the agent (defaults to env config)

    Returns:
        Dictionary containing the agent's response
    """
    if agent_config is None:
        agent_config = AgentConfig.from_env()

    # Prepare initial state using standard MessagesState pattern
    initial_state: TodoAgentState = {
        "messages": [HumanMessage(content=user_input)],
        "user_id": user_id,
        "todo_results": None,
        "memory_context": None,
        "processing_complete": False,
    }

    # Prepare configuration
    config: RunnableConfig = {
        "configurable": {"agent_config": agent_config, "user_id": user_id},
        "recursion_limit": 10,
    }

    try:
        # Run the graph
        result = await graph.ainvoke(initial_state, config)
        return result
    except Exception as e:
        return {
            "messages": [HumanMessage(content=f"Error running todo agent: {str(e)}")],
            "error": str(e),
        }


# Example usage function
async def example_usage():
    """Example of how to use the todo agent with improved patterns."""
    # Create configuration from environment
    config = AgentConfig.from_env()

    # Test cases demonstrating different capabilities
    test_cases = [
        "Add buy groceries and call mom to my todo list",
        "I need to finish the report by Friday and schedule dentist appointment",
        "Create todos for: 1. Review code 2. Send email to client 3. Update docs",
        "Show me my current todos",
        "What todos did I create today?",
        "Mark todo_0001 as completed",
    ]

    user_id = "test_user_123"

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Input: {test_input}")

        result = await run_todo_agent(test_input, user_id, config)

        # Extract response from messages
        response_text = "No response"
        if result.get("messages"):
            latest_msg = result["messages"][-1]
            if hasattr(latest_msg, "content"):
                response_text = latest_msg.content

        print(f"Response: {response_text}")

        if result.get("error"):
            print(f"Error: {result['error']}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_usage())
