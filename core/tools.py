"""Function tools for the agent."""
from agents import function_tool
from .rag import (
    query_chromadb, 
    aggregate_by_book, 
    rank_books, 
    generate_query_embedding
)
from .database import search_books_by_author, get_all_books
from .memory import get_persona

@function_tool
def semantic_search(query: str) -> str:
    """
    Search for book content/excerpts matching a conceptual query.
    Best for: Finding books about specific themes, plots, or emotions.
    """
    print(f"🔍 [TOOL] semantic_search: {query}")
    query_emb = generate_query_embedding(query)
    results = query_chromadb(query_emb)
    book_scores = aggregate_by_book(results)
    ranked = rank_books(book_scores)
    
    if not ranked:
        return "No content matching that query was found."
    
    output = "📚 Recommended Books:\n\n"
    for i, b in enumerate(ranked, 1):
        output += f"📖 **Book {i}: {b['book_name']}**\n"
        output += f"   👤 Author: {b['author_name']}\n"
        output += f"   📅 Release Date: {b['release_date']}\n"
        output += f"   🎯 Relevance Score: {b['score']:.2f}\n"
        output += f"   📝 Sample Excerpt: {b['best_chunk'][:200]}...\n\n"
    return output

@function_tool
def metadata_search(author: str = None, title: str = None) -> str:
    """
    Search for books by specific author or title in the formal database.
    Best for: Finding all books by a specific writer or checking release dates.
    """
    print(f"📇 [TOOL] metadata_search: author={author}, title={title}")
    if author:
        books = search_books_by_author(author)
    else:
        books = get_all_books()
        
    if not books:
        return "No books found matching those criteria."
        
    output = "📚 Book Database Results:\n\n"
    for b in books:
        output += f"📖 **{b['book_name']}**\n"
        output += f"   👤 Author: {b['author_name']}\n"
        output += f"   📅 Released: {b['release_date']}\n"
        output += f"   🆔 Book ID: {b['book_id']}\n\n"
    return output

@function_tool
def get_user_persona(user_id: str = "default_user") -> str:
    """
    Retrieve the user's recorded interests and past preferences.
    Best for: Tailoring recommendations to the user's taste.
    """
    print(f"👤 [TOOL] get_user_persona: {user_id}")
    return get_persona(user_id)
