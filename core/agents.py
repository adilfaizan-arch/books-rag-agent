"""Agent configuration and execution using OpenAI Agents SDK."""
from agents import Agent, Runner
from .tools import semantic_search, metadata_search, get_user_persona
from .memory import save_interaction, update_persona

def create_agent():
    """Create and configure the RAG agent with a memory-aware multi-tool workflow."""
    
    agent = Agent(
        name="Book Personal Assistant",
        model="gpt-4o-mini",
        instructions="""You are a personalized book recommendation assistant. You have a four-tool system to help you provide the best advice.

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
- **Persona Updates**: Always invoke `update_persona` in parallel when new user preferences are inferred from the query.
- **Structured Details**: Always present books with clear, structured formatting including title, author, date, and relevant excerpts.
- **Friendly Tone**: Be an enthusiastic librarian who loves connecting people with great stories.
- **Updation of user persona**: This is must do activity when ever you feel the user's preferences have changed or evolved keep the persona updated.  It is a must process.
""",
        tools=[get_user_persona, metadata_search, semantic_search, update_persona]
    )
    
    return agent

def run_query(query: str, user_id: str = "default_user", history: list = None):
    """Run a query through the agent and save the interaction to memory.
    
    Args:
        query: User question or search query
        user_id: The ID of the user (defaults to 'default_user')
        history: List of previous interactions (optional)
        
    Returns:
        Agent response
    """
    agent = create_agent()

    # Prepare the context with history
    history_context = "\n".join([f"User: {h['query']}\nAssistant: {h['response']}" for h in history]) if history else ""
    full_query = f"PAST HISTORY ::\n{history_context}\n\n\nUser Query: {query}"

    # Run agent with query context
    result = Runner.run_sync(agent, f"[USER_ID: {user_id}] {full_query}")

    # Save the interaction to memory layer
    save_interaction(user_id, query, result.final_output)

    # Update persona in parallel if the query suggests it
    # if "book" in query.lower() or "recommend" in query.lower():
    #     update_text = f"User asked: '{query}'"
    #     update_persona(user_id, update_text)

    return result.final_output
