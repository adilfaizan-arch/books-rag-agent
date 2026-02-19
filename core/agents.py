"""Agent configuration and execution using OpenAI Agents SDK."""
from pathlib import Path
from agents import Agent, Runner
from agents.memory import SQLiteSession, OpenAIResponsesCompactionSession
from .tools import semantic_search, metadata_search, get_user_persona
from .memory import update_persona

# Single database for all user sessions and personas
SESSIONS_DB = Path("memory/user_sessions.db")
SESSIONS_DB.parent.mkdir(parents=True, exist_ok=True)

def create_agent(user_id: str = "default_user"):
    """Create and configure the RAG agent with a memory-aware multi-tool workflow."""
    
    agent = Agent(
        name="Book Personal Assistant",
        model="gpt-4o-mini",
        instructions=f"""You are a personalized book recommendation assistant for the user: {user_id}. You have a four-tool system to help you provide the best advice.

### YOUR USER:
The current user is: {user_id}. Always use this ID when calling persona tools.

### YOUR WORKFLOW:
Workflow should be according to the user's query and needs:
If user mentions any author, writer, or book name or year of publication, then you should use metadata search to find the book and its details.
If user is looking for specific themes, emotions, or plot points, use semantic search to find relevant excerpts.
If user asks for recommendations without specific details, start by checking their persona to understand their preferences and then use a combination of metadata and semantic search to find the best matches.
If the query suggests new information about the user's preferences (e.g., asking for a specific author or genre), invoke the `update_persona` tool in parallel to append this information to their persona.
And after observing the users queries and interations please update the persona of the user in the memory layer by invoking the update_persona tool in parallel.

### YOUR GOAL:
Provide highly relevant, conversational recommendations with structured book details.

### RESPONSE FORMAT:
When presenting books, always include:
- 📖 Book title (bold)
- 👤 Author name
- 📅 Release date
- 🎯 Relevance score (for semantic search)
- 📝 Brief excerpt or description

### GUIDELINES:
- **Conversational Proactivity**: If a user's request is broad (e.g., "recommend me a book"), ask clarifying questions about genres, moods, or specific authors they enjoy before giving a final list.
- **Metadata Precision**: If a user asks for books from a specific year, author, or title, use `metadata_search` to get exact results from the database.
- **Persona Updates**: Always invoke `update_persona` in parallel when new user preferences are inferred from the query. You MUST identify key interest categories (e.g., ["Sci-Fi", "Philosophy"]) and pass them as a list to the 'interests' argument. Use user_id="{user_id}".
- **Structured Details**: Always present books with clear, structured formatting including title, author, date, and relevant excerpts.
- **Friendly Tone**: Be an enthusiastic librarian who loves connecting people with great stories.
- **Updation of user persona**: This is must do activity when ever you feel the user's preferences have changed or evolved keep the persona updated. It is a must process.
""",
        tools=[get_user_persona, metadata_search, semantic_search, update_persona]
    )
    
    return agent

def create_session(user_id: str):
    """Create a persistent session with automatic compaction for a user.
    
    All user sessions are stored in the same SESSIONS_DB file.
    The SDK handles separation using the session_id.
    
    Args:
        user_id: Unique identifier for the user
        
    Returns:
        OpenAIResponsesCompactionSession wrapping SQLiteSession
    """
    # Use the single unified database for all users
    sqlite_session = SQLiteSession(
        session_id=user_id,
        db_path=str(SESSIONS_DB)
    )
    
    # Define a custom trigger to compact more frequently (after 5 non-user messages)
    def should_trigger(cache):
        items=cache.get("session_items", [])
        non_user_items=[item for item in items if item.get("role") != "user"]
        # print("Cache contents:", cache)
        return len(non_user_items) >= 5

    # Wrap with compaction session for automatic conversation summarization
    compaction_session = OpenAIResponsesCompactionSession(
        session_id=user_id,
        underlying_session=sqlite_session,
        model="gpt-4o-mini",
        compaction_mode="auto",
        should_trigger_compaction=should_trigger
    )
    
    return compaction_session

def run_query(query: str, user_id: str = "default_user"):
    """Run a query through the agent with persistent session management.
    
    Args:
        query: User question or search query
        user_id: The ID of the user (defaults to 'default_user')
        
    Returns:
        Agent response
    """
    agent = create_agent(user_id)
    session = create_session(user_id)
    
    # Run agent with persistent session
    # The session automatically manages conversation history and compaction
    result = Runner.run_sync(agent, query, session=session)
    
    return result.final_output
