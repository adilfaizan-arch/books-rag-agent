"""Core RAG pipeline package."""
from .rag import index_books, retrieve_and_aggregate, generate_embedding, generate_query_embedding
from .tools import semantic_search, metadata_search, get_user_persona
from .agents import create_agent, run_query, create_session
from .database import (
    init_database, add_book, get_book, get_all_books, search_books_by_author
)
from .memory import get_persona, update_persona, get_persona_from_db, save_persona_update_to_db

__all__ = [
    'generate_embedding',
    'generate_query_embedding',
    'index_books',
    'retrieve_and_aggregate',
    'semantic_search',
    'metadata_search',
    'get_user_persona',
    'create_agent',
    'run_query',
    'create_session',
    'init_database',
    'add_book',
    'get_book',
    'get_all_books',
    'search_books_by_author',
    'get_persona_from_db',
    'save_persona_update_to_db',
    'get_persona',
    'update_persona'
]
