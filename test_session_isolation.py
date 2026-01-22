#!/usr/bin/env python3
"""
Test script for verifying user session isolation
Run this after the Render deployment is complete
"""

import requests
import time

# Your Render API URL
BASE_URL = "https://langchain-rag-api.onrender.com"

def test_user_isolation():
    print("=" * 60)
    print("Testing User Session Isolation")
    print("=" * 60)
    
    # Step 1: User A asks about LangChain
    print("\n[1] User A: 'What is LangChain?'")
    resp = requests.post(f"{BASE_URL}/chat/conversation", json={
        "question": "What is LangChain?",
        "session_id": "user-A"
    })
    print(f"Response: {resp.json()['answer'][:200]}...")
    
    time.sleep(1)
    
    # Step 2: User A asks a follow-up (should have context)
    print("\n[2] User A: 'Tell me more about it' (should reference LangChain)")
    resp = requests.post(f"{BASE_URL}/chat/conversation", json={
        "question": "Tell me more about it",
        "session_id": "user-A"
    })
    answer_a = resp.json()['answer']
    print(f"Response: {answer_a[:200]}...")
    
    time.sleep(1)
    
    # Step 3: User B asks same follow-up (should NOT have context from User A)
    print("\n[3] User B: 'Tell me more about it' (should NOT have LangChain context)")
    resp = requests.post(f"{BASE_URL}/chat/conversation", json={
        "question": "Tell me more about it",
        "session_id": "user-B"
    })
    answer_b = resp.json()['answer']
    print(f"Response: {answer_b[:200]}...")
    
    # Verify isolation
    print("\n" + "=" * 60)
    print("VERIFICATION:")
    if "langchain" in answer_a.lower() and "langchain" not in answer_b.lower():
        print("✅ SUCCESS: User sessions are properly isolated!")
    else:
        print("⚠️  Check manually - User A should mention LangChain, User B should not")
    print("=" * 60)
    
    # Step 4: List all sessions
    print("\n[4] Listing all active sessions:")
    resp = requests.get(f"{BASE_URL}/chat/sessions")
    print(resp.json())
    
    # Step 5: Clear User A's session
    print("\n[5] Clearing User A's session...")
    resp = requests.delete(f"{BASE_URL}/chat/conversation/user-A")
    print(resp.json())
    
    # Step 6: Verify User A's context is cleared
    print("\n[6] User A: 'Tell me more about it' (should have NO context now)")
    resp = requests.post(f"{BASE_URL}/chat/conversation", json={
        "question": "Tell me more about it",
        "session_id": "user-A"
    })
    print(f"Response: {resp.json()['answer'][:200]}...")

if __name__ == "__main__":
    # First check if the API is up
    print("Checking API health...")
    try:
        resp = requests.get(f"{BASE_URL}/health")
        print(f"Health: {resp.json()}")
        test_user_isolation()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the Render deployment is complete first!")
