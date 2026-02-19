import os
import asyncio
import random
import numpy as np
import argparse
from dotenv import load_dotenv
from core import run_query
import traceback
import json
from pathlib import Path

# Load environment variables from .env
load_dotenv()

# Explicitly disable LangSmith/LangChain tracing to prevent rate limit errors
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
if "LANGSMITH_API_KEY" in os.environ:
    del os.environ["LANGSMITH_API_KEY"]

# Set random seeds
random.seed(42)
np.random.seed(42)

def run_interactive_loop(user_id: str):
    """Run an interactive chat loop in the terminal."""
    print(f"\n{'='*60}")
    print(f"INTERACTIVE MODE: Researching as {user_id}")
    print(f"Type 'exit' or 'quit' to end. Happy researching!")
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
            response = asyncio.run(run_query(query, user_id=user_id))
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
    parser = argparse.ArgumentParser(description="HICSS Research Assistant via MCP")
    parser.add_argument("--user", type=str, default="researcher_1", help="User ID for the session")
    args = parser.parse_args()

    print("=" * 60)
    print("HICSS RESEARCH ASSISTANT: Academic Paper Search via MCP")
    print("=" * 60)
    
    # Start the interactive research loop immediately
    run_interactive_loop(args.user)

if __name__ == "__main__":
    main()
