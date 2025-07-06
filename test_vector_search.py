#!/usr/bin/env python3
"""
Test script for Qdrant vector search with Ollama embeddings.
Demonstrates the complete workflow of getting embeddings and searching.
"""

import json
import requests
import sys
from typing import List, Dict, Any

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
QDRANT_BASE_URL = "http://localhost:6333"
EMBEDDING_MODEL = "bge-m3:latest"
COLLECTION_NAME = "todo_memories"

def get_embedding(text: str) -> List[float]:
    """Get embedding vector from Ollama."""
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    payload = {
        "model": EMBEDDING_MODEL,
        "prompt": text
    }
    
    print(f"Getting embedding for: '{text}'")
    response = requests.post(url, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"Ollama embedding failed: {response.status_code} - {response.text}")
    
    data = response.json()
    embedding = data.get("embedding")
    
    if not embedding:
        raise Exception("No embedding returned from Ollama")
    
    print(f"✅ Got embedding vector with {len(embedding)} dimensions")
    return embedding

def search_qdrant(query_vector: List[float], limit: int = 10, score_threshold: float = 0.5) -> Dict[str, Any]:
    """Search Qdrant using vector similarity."""
    url = f"{QDRANT_BASE_URL}/collections/{COLLECTION_NAME}/points/search"
    
    payload = {
        "vector": query_vector,
        "limit": limit,
        "score_threshold": score_threshold,
        "with_payload": True,
        "with_vector": False
    }
    
    print(f"Searching Qdrant collection '{COLLECTION_NAME}' with {len(query_vector)}D vector...")
    response = requests.post(url, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"Qdrant search failed: {response.status_code} - {response.text}")
    
    return response.json()

def check_qdrant_collection() -> Dict[str, Any]:
    """Check if the collection exists and get its info."""
    url = f"{QDRANT_BASE_URL}/collections/{COLLECTION_NAME}"
    response = requests.get(url)
    
    if response.status_code == 404:
        print(f"❌ Collection '{COLLECTION_NAME}' not found")
        return {}
    elif response.status_code != 200:
        raise Exception(f"Failed to check collection: {response.status_code} - {response.text}")
    
    return response.json()

def test_vector_search():
    """Main test function."""
    print("🔍 Vector Search Test for Qdrant + Ollama")
    print("=" * 50)
    
    # Test query
    query_text = "Show me all the tasks related "
    
    try:
        # 1. Check Qdrant collection
        print("\n1. Checking Qdrant collection...")
        collection_info = check_qdrant_collection()
        if collection_info:
            result = collection_info.get("result", {})
            config = result.get("config", {})
            status = result.get("status", "unknown")
            points_count = result.get("points_count", 0)
            
            print(f"✅ Collection exists: {COLLECTION_NAME}")
            print(f"   Status: {status}")
            print(f"   Points count: {points_count}")
            print(f"   Vector size: {config.get('params', {}).get('vectors', {}).get('size', 'unknown')}")
        else:
            print(f"⚠️  Collection '{COLLECTION_NAME}' doesn't exist yet")
            print("   This is normal if you haven't created any memories yet")
        
        # 2. Get embedding from Ollama
        print(f"\n2. Getting embedding from Ollama...")
        try:
            embedding = get_embedding(query_text)
        except Exception as e:
            print(f"❌ Failed to get embedding: {e}")
            print("   Make sure Ollama is running and the model is available:")
            print(f"   ollama pull {EMBEDDING_MODEL}")
            return False
        
        # 3. Search Qdrant (only if collection exists and has points)
        if collection_info and collection_info.get("result", {}).get("points_count", 0) > 0:
            print(f"\n3. Searching Qdrant...")
            try:
                search_results = search_qdrant(embedding, limit=5, score_threshold=0.3)
                
                results = search_results.get("result", [])
                print(f"✅ Search completed - found {len(results)} results")
                
                if results:
                    print("\n📋 Search Results:")
                    for i, result in enumerate(results, 1):
                        score = result.get("score", 0)
                        payload = result.get("payload", {})
                        point_id = result.get("id")
                        
                        print(f"\n  {i}. Score: {score:.4f} (ID: {point_id})")
                        for key, value in payload.items():
                            if isinstance(value, str) and len(value) > 100:
                                value = value[:100] + "..."
                            print(f"     {key}: {value}")
                else:
                    print("   No results found above the score threshold")
                    
            except Exception as e:
                print(f"❌ Qdrant search failed: {e}")
                return False
        else:
            print(f"\n3. Skipping Qdrant search - no data in collection")
            print("   Add some memories first using the todo agent")
        
        # 4. Show raw curl commands
        print(f"\n4. Equivalent curl commands:")
        print(f"\n   # Get embedding:")
        embedding_curl = f"""curl -X POST {OLLAMA_BASE_URL}/api/embeddings \\
  -H "Content-Type: application/json" \\
  -d '{{"model": "{EMBEDDING_MODEL}", "prompt": "{query_text}"}}'"""
        print(f"   {embedding_curl}")
        
        if collection_info and collection_info.get("result", {}).get("points_count", 0) > 0:
            print(f"\n   # Search Qdrant (using the embedding vector):")
            search_curl = f"""curl -X POST {QDRANT_BASE_URL}/collections/{COLLECTION_NAME}/points/search \\
  -H "Content-Type: application/json" \\
  -d '{{"vector": [<embedding_vector>], "limit": 5, "score_threshold": 0.3, "with_payload": true}}'"""
            print(f"   {search_curl}")
        
        print(f"\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_vector_search()
    sys.exit(0 if success else 1) 