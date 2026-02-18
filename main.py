import os
import random
import numpy as np
import argparse
from dotenv import load_dotenv
from langsmith.integrations.openai_agents_sdk import OpenAIAgentsTracingProcessor
from agents import set_trace_processors
from core import index_books, run_query
from core.database import init_database
import traceback
from core.memory import get_persona
import json
from pathlib import Path

# Load environment variables from .env
load_dotenv()

# Disable LangSmith tracing by commenting out the environment variables
# os.environ["LANGSMITH_TRACING"] = "true"
# os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGSMITH_PROJECT"] = "books"

# Comment out the set_trace_processors setup
# try:
#     from agents import set_trace_processors
#     set_trace_processors([OpenAIAgentsTracingProcessor()])
# except ImportError:
#     pass

# Set random seeds
random.seed(42)
np.random.seed(42)

def run_interactive_loop(user_id: str):
    """Run an interactive chat loop in the terminal."""
    print(f"\n{'='*60}")
    print(f"INTERACTIVE MODE: Chatting as {user_id}")
    print(f"Type 'exit' or 'quit' to end. Happy reading!")
    print(f"{'='*60}\n")

    while True:
        try:
            query = input(f"You ({user_id}) > ")
            if query.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break

            if not query.strip():
                continue

            print("\nThinking...")
            # Session history is automatically managed by SQLiteSession
            response = run_query(query, user_id=user_id)
            print(f"\nAssistant: {response}\n")
            print("-" * 30)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print("\nError occurred:")
            traceback.print_exc()

def main():
    """Main execution flow."""
    parser = argparse.ArgumentParser(description="Personalized Books RAG")
    parser.add_argument("--demo", action="store_true", help="Run the automated demo queries")
    parser.add_argument("--user", type=str, default="user_123", help="User ID for the session")
    parser.add_argument("--skip-index", action="store_true", help="Skip indexing step entirely")
    parser.add_argument("--force-index", action="store_true", help="Force re-indexing even if books exist")
    args = parser.parse_args()

    print("=" * 60)
    print("PERSONALIZED RAG: Interactive Library Assistant")
    print("=" * 60)
    
    # Step 0: Initialize database
    init_database()
    
    # Step 1: Index books
    if not args.skip_index:
        print(f"\n[1] Indexing books (force={args.force_index})...")
        index_books(force=args.force_index)
    
    # Step 2: Run queries
    if args.demo:
        print(f"\n[2] Running demo queries for user: {args.user}")
        queries = [
            "I really love adventure and exploring unknown places.",
            "Recommend me something based on what I like.",
            "Who wrote the adventure books you found earlier?"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n--- Interaction {i} ---")
            print(f"User: {query}")
            response = run_query(query, user_id=args.user)
            print(f"\nAssistant: {response}")
            print("-" * 30)
    else:
        run_interactive_loop(args.user)

if __name__ == "__main__":
    main()
