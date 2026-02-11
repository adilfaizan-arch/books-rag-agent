"""Memory layer for tracking user interests and chat history."""

import os
import json
import re
from pathlib import Path
from datetime import datetime
from langsmith import traceable
from openai import OpenAI
from agents import function_tool

MEMORY_DIR = Path("memory")

# Global OpenAI client, initialized lazily
_openai_client = None

def get_openai_client():
    """Lazily initialize the OpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client

@traceable(name="Save Interaction", run_type="tool")
def save_interaction(user_id: str, query: str, response: str):
    """Save user interaction and update persona keywords."""
    if not MEMORY_DIR.exists():
        MEMORY_DIR.mkdir()

    user_file = MEMORY_DIR / f"{user_id}.json"
    
    # Load existing memory
    if user_file.exists():
        with open(user_file, 'r') as f:
            memory = json.load(f)
    else:
        memory = {"history": [], "interests": []}

    # Add to history
    memory["history"].append({
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "response": response
    })

    # Simple keyword extraction (for interests)
    # In a real app, you'd use LLM to extract interests
    # new_interests = extract_interests(query)
    # for interest in new_interests:
    #     if interest.lower() not in [i.lower() for i in memory["interests"]]:
    #         memory["interests"].append(interest)

    # Save memory
    with open(user_file, 'w') as f:
        json.dump(memory, f, indent=2)
    
    print(f"✓ Interaction saved for {user_id}.")

def extract_interests(text: str):
    """Extract potential interest keywords from text."""
    # List of common topics in our books
    topics = ["adventure", "mystery", "love", "science", "horror", "detective", "nature", "travel", "philosophy", "history"]
    found = []
    for topic in topics:
        if re.search(rf"\b{topic}\b", text, re.I):
            found.append(topic.capitalize())
    return found

@traceable(name="Get User Persona", run_type="tool")
def get_persona(user_id: str):
    """Retrieve user interests and persona from memory."""
    user_file = MEMORY_DIR / f"{user_id}.json"
    if not user_file.exists():
        return "New user. No specific interests recorded yet."
    
    with open(user_file, 'r') as f:
        memory = json.load(f)
    
    interests = memory.get("interests", [])
    if not interests:
        return "User has interacted but no specific interests identified yet."
    
    return f"User is interested in: {', '.join(interests)}. Keep these preferences in mind for recommendations."

@function_tool
def update_persona(user_id: str, update_text: str):
    """
    Update the user's persona with new information on the basis of user chat and interations with the agent.
    """
    if not MEMORY_DIR.exists():
        MEMORY_DIR.mkdir()

    user_file = MEMORY_DIR / f"{user_id}.json"

    # Load existing memory
    if user_file.exists():
        with open(user_file, 'r') as f:
            memory = json.load(f)
    else:
        memory = {"history": [], "interests": []}

    # Append the update text to the persona
    if "persona_updates" not in memory:
        memory["persona_updates"] = []

    memory["persona_updates"].append({
        "timestamp": datetime.now().isoformat(),
        "update": update_text
    })

    # Save memory
    with open(user_file, 'w') as f:
        json.dump(memory, f, indent=2)

    print(f"✓ Persona updated for {user_id}: {update_text}")
