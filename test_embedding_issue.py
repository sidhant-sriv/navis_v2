#!/usr/bin/env python3
"""Test to isolate the embedding issue between user and todo nodes."""

import sys
import traceback
from src.agent.config import AgentConfig
from src.agent.memory_adapters import create_memory_adapter
from src.agent.tools import UnifiedTodoStorage
from src.agent.models import TodoItem, UserInfo, UserInfoType

def test_user_embedding():
    """Test embedding for user nodes."""
    print("=" * 60)
    print("Testing User Node Embedding (via MemoryAdapter)")
    print("=" * 60)
    
    try:
        config = AgentConfig.from_env()
        adapter = create_memory_adapter(config)
        
        # Try to store a user info
        user_info = UserInfo(
            id="test_user_001",
            user_id="test_user",
            info_type=UserInfoType.PERSONAL,
            content="Test user information for embedding",
            relevance_score=0.9,
            tags=["test"]
        )
        
        print("Attempting to store user info...")
        result = adapter.store_user_info(user_info)
        
        if result:
            print("✅ User node embedding SUCCESSFUL")
            return True
        else:
            print("❌ User node embedding FAILED (but no exception)")
            return False
            
    except Exception as e:
        print(f"❌ User node embedding FAILED with exception:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        traceback.print_exc()
        return False


def test_todo_embedding():
    """Test embedding for todo nodes."""
    print("\n" + "=" * 60)
    print("Testing Todo Node Embedding (via UnifiedTodoStorage)")
    print("=" * 60)
    
    try:
        config = AgentConfig.from_env()
        storage = UnifiedTodoStorage(config)
        
        # Try to store a todo
        todo = TodoItem(
            id="test_todo_001",
            title="Test todo for embedding",
            description="This is a test todo to check embedding functionality",
            priority="medium",
            tags=["test"]
        )
        
        print("Attempting to store todo...")
        result = storage.store_todo(todo, "test_user")
        
        if result:
            print("✅ Todo node embedding SUCCESSFUL")
            return True
        else:
            print("❌ Todo node embedding FAILED (but no exception)")
            return False
            
    except Exception as e:
        print(f"❌ Todo node embedding FAILED with exception:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        traceback.print_exc()
        return False


def test_direct_ollama_embedding():
    """Test direct Ollama embedding API call."""
    print("\n" + "=" * 60)
    print("Testing Direct Ollama Embedding API")
    print("=" * 60)
    
    try:
        import requests
        
        url = "http://localhost:11434/api/embeddings"
        payload = {
            "model": "bge-m3:latest",
            "prompt": "Test embedding from direct API call"
        }
        
        print(f"Calling {url}...")
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            embedding = response.json().get("embedding", [])
            print(f"✅ Direct Ollama API call SUCCESSFUL")
            print(f"   Got embedding with {len(embedding)} dimensions")
            return True
        else:
            print(f"❌ Direct Ollama API call FAILED")
            print(f"   Status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Direct Ollama API call FAILED with exception:")
        print(f"   Error type: {type(e).__name__}")
        print(f"   Error message: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Run all tests to isolate the issue."""
    print("\n🔍 EMBEDDING ISSUE DIAGNOSTIC TEST")
    print("This will test embeddings for user nodes, todo nodes, and direct API calls")
    print("\n")
    
    results = {
        "direct_ollama": test_direct_ollama_embedding(),
        "user_nodes": test_user_embedding(),
        "todo_nodes": test_todo_embedding(),
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
    
    print("\n" + "=" * 60)
    
    if results["direct_ollama"] and results["user_nodes"] and not results["todo_nodes"]:
        print("\n🔍 DIAGNOSIS:")
        print("Direct Ollama works ✅")
        print("User nodes work ✅")
        print("Todo nodes fail ❌")
        print("\nThis suggests the issue is specific to how todo nodes are stored,")
        print("not with the embedding service itself.")
    elif not results["direct_ollama"]:
        print("\n🔍 DIAGNOSIS:")
        print("The Ollama embedding service is not responding correctly.")
        print("Please check if Ollama is running and the model is available:")
        print("  ollama list | grep bge-m3")
        print("  ollama pull bge-m3:latest")
    else:
        print("\n🔍 DIAGNOSIS:")
        print("Results are mixed. Please review the detailed output above.")
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
