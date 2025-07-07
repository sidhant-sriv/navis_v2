"""Demo script for the LangGraph Todo Agent with Mem0 and Qdrant.

Prerequisites:
1. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant
2. Start Ollama: ollama serve
3. Pull required models: ollama pull llama3.1:8b && ollama pull nomic-embed-text
4. Install dependencies: pip install -e .

Usage:
    python demo.py
"""

import asyncio
import os
from src.agent.graph import run_todo_agent
from src.agent.config import AgentConfig


async def demo_todo_agent():
    """Demonstrate the todo agent capabilities."""

    print("🚀 LangGraph Todo Agent Demo")
    print("=" * 50)

    # Load configuration from environment
    config = AgentConfig.from_env()
    user_id = "demo_user"

    # Demo scenarios
    scenarios = [
        {
            "name": "Creating Multiple Todos",
            "input": "I need to buy groceries, call my dentist to schedule an appointment, and finish the quarterly report by Friday",
            "description": "Tests extracting multiple todo items from natural language",
        },
        {
            "name": "Structured Todo Creation",
            "input": "Create todos for: 1. Review pull request 2. Update documentation 3. Send project update email",
            "description": "Tests parsing numbered list format",
        },
        {
            "name": "Priority and Date Extraction",
            "input": "Add urgent task: fix critical bug ASAP, and low priority: organize desk when possible",
            "description": "Tests priority detection and scheduling",
        },
        {
            "name": "Listing Todos",
            "input": "Show me all my current todos",
            "description": "Tests memory retrieval and todo listing",
        },
        {
            "name": "Memory Search",
            "input": "What tasks did I create related to work?",
            "description": "Tests semantic memory search",
        },
        {
            "name": "Todo Completion",
            "input": "Mark my grocery shopping task as completed",
            "description": "Tests todo completion tracking",
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n📝 Scenario {i}: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        print(f"Input: '{scenario['input']}'")
        print("-" * 50)

        try:
            result = await run_todo_agent(
                user_input=scenario["input"], user_id=user_id, agent_config=config
            )

            # Extract response from messages (same as in graph.py example_usage)
            response_text = "No response generated"
            if result.get("messages"):
                latest_msg = result["messages"][-1]
                if hasattr(latest_msg, "content"):
                    response_text = latest_msg.content

            print(f"Response:\n{response_text}")

            if result.get("error"):
                print(f"⚠️  Error: {result['error']}")

        except Exception as e:
            print(f"❌ Failed to run scenario: {str(e)}")

        print("\n" + "=" * 50)

        # Pause between scenarios for readability
        await asyncio.sleep(1)

    print("\n✅ Demo completed!")
    print("\nKey Features Demonstrated:")
    print("• Multiple todo extraction from natural language")
    print("• Priority and due date detection")
    print("• Persistent memory with Mem0 and Qdrant")
    print("• Semantic search across past interactions")
    print("• Local LLM processing with Ollama")


def check_prerequisites():
    """Check if required services are running."""
    print("🔍 Checking prerequisites...")

    # Check environment variables
    required_env_vars = []
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("You can set defaults or these will use default values.")

    # Test Qdrant connection
    try:
        import qdrant_client

        client = qdrant_client.QdrantClient("localhost", port=6333)
        client.get_collections()
        print("✅ Qdrant is running on localhost:6333")
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        print("Start Qdrant with: docker run -p 6333:6333 qdrant/qdrant")
        return False

    # Test Ollama connection
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running on localhost:11434")
        else:
            print("❌ Ollama responded with error")
            return False
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("Start Ollama with: ollama serve")
        return False

    return True


def interactive_mode():
    """Run interactive mode for manual testing."""
    print("\n🎮 Interactive Mode")
    print("Enter todo-related requests (type 'quit' to exit):")
    print("Examples:")
    print("- 'Add wash car and buy milk to my todos'")
    print("- 'Show my current tasks'")
    print("- 'What did I add yesterday?'")

    user_id = input(
        "\nEnter your user ID (or press Enter for 'interactive_user'): "
    ).strip()
    if not user_id:
        user_id = "interactive_user"

    config = AgentConfig.from_env()

    while True:
        try:
            user_input = input(f"\n[{user_id}] > ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue

            print("🤔 Processing...")
            result = asyncio.run(run_todo_agent(user_input, user_id, config))

            # Extract response from messages (same as in graph.py example_usage)
            response_text = "No response generated"
            if result.get("messages"):
                latest_msg = result["messages"][-1]
                if hasattr(latest_msg, "content"):
                    response_text = latest_msg.content

            print(f"\n🤖 Assistant: {response_text}")

            if result.get("error"):
                print(f"⚠️  Error: {result['error']}")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


def main():
    """Main entry point."""
    print("LangGraph Todo Agent with Mem0, Qdrant & Ollama")
    print("================================================")

    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return

    print("\nChoose a mode:")
    print("1. Run demo scenarios")
    print("2. Interactive mode")

    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()

        if choice == "1":
            asyncio.run(demo_todo_agent())
            break
        elif choice == "2":
            interactive_mode()
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()
