"""Agent configuration and execution using OpenAI Agents SDK."""
import os
import json

# Explicitly disable LangSmith/LangChain tracing to prevent rate limit errors
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
if "LANGSMITH_API_KEY" in os.environ:
    del os.environ["LANGSMITH_API_KEY"]

from pathlib import Path
from agents import Agent, Runner
from agents.memory import SQLiteSession, OpenAIResponsesCompactionSession
# Single database for all user sessions and personas
SESSIONS_DB = Path("memory/user_sessions.db")
SESSIONS_DB.parent.mkdir(parents=True, exist_ok=True)

def create_agent(user_id: str = "default_user"):
    """Create and configure the RAG agent using the remote HICSS MCP server for academic research."""
    
    agent = Agent(
        name="HICSS Research Assistant",
        model="gpt-4o-mini",
        instructions=f"""You are a specialized Research Assistant for the user: {user_id}. You have access to the HICSS SmartSearch MCP server, which contains a vast collection of academic papers.

### YOUR GOAL:
Find and summarize relevant research papers, authors, and academic insights using the connected MCP tools.


### YOUR WORKFLOW:
- When a user asks a research question, invoke the appropriate MCP tool immediately.
- If the user specifies a year or range, prioritize `search_attribute`.
- Provide high-quality, professional summaries of the findings.

### RESPONSE FORMAT:
Always present papers with:
- 📖 **Title** (bold)
- 👤 Author(s)
- 📅 Publication Date/Year
- 📝 Abstract or Description 

### GUIDELINES:
- **Conversational Proactivity**: Ask clarifying questions if the research topic is too broad.
- **Accuracy**: Rely strictly on the data returned by the HICSS server.
- **Professional Tone**: Be a helpful, knowledgeable peer in the research process.
""",
        tools=[]  # All tools come from remote HICSS MCP discovery
    )
    
    return agent

def create_session(user_id: str):
    """Create a persistent session with automatic compaction for a user."""
    sqlite_session = SQLiteSession(
        session_id=user_id,
        db_path=str(SESSIONS_DB)
    )
    
    def should_trigger(cache):
        items = cache.get("session_items", [])
        non_user_items = [item for item in items if item.get("role") != "user"]
        return len(non_user_items) >= 5

    compaction_session = OpenAIResponsesCompactionSession(
        session_id=user_id,
        underlying_session=sqlite_session,
        model="gpt-4o-mini",
        compaction_mode="auto",
        should_trigger_compaction=should_trigger
    )
    
    return compaction_session


async def run_query(query: str, user_id: str = "default_user"):
    """Run a query through the agent with persistent session management.
    
    Connects to the HICSS SmartSearch MCP server via SSE and passes it
    to the agent.
    
    Args:
        query: User question or search query
        user_id: The ID of the user (defaults to 'default_user')
        
    Returns:
        Agent response
    """
    from agents.mcp import MCPServerSse

    agent = create_agent(user_id)
    session = create_session(user_id)

    async with MCPServerSse(
        name="HICSS SmartSearch",
        params={
            "url": "https://hicss-smartsearch.arbisoft.com/",
            "timeout": 30,
        },
        client_session_timeout_seconds=30,
        cache_tools_list=True,
    ) as mcp_server:
        agent.mcp_servers = [mcp_server]
        
        
        # Use streaming to intercept tool calls in real-time
        streamed_result = Runner.run_streamed(agent, query, session=session)
        
        async for event in streamed_result.stream_events():
            # Intercept 'tool_called' events to show the user what the agent is doing
            if hasattr(event, "name") and event.name == "tool_called":
                tool_item = getattr(event, "item", None)
                if tool_item and hasattr(tool_item, "raw_item"):
                    tool_call = tool_item.raw_item
                    # The tool name might be in tool_call.name or tool_call.function.name
                    tool_name = getattr(tool_call, "name", None)
                    if not tool_name and hasattr(tool_call, "function"):
                        tool_name = getattr(tool_call.function, "name", "mcp_tool")
                    
                    print(f"\n[RESEARCHING] Calling tool: {tool_name or 'mcp_tool'}...")
                    
                    # Attempt to parse and print arguments
                    try:
                        args = getattr(tool_call, "arguments", None)
                        if args is None and hasattr(tool_call, "function"):
                            args = getattr(tool_call.function, "arguments", None)
                        
                        if isinstance(args, str):
                            args = json.loads(args)
                        
                        if args:
                            print(f"  🔍 Parameters: {json.dumps(args, indent=2)}")
                    except Exception:
                        pass
        
        # The final result is available on the streamed_result object after the stream ends
        result = streamed_result.final_output

    return result
