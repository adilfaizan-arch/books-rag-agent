import torch
import random
import os
import PyPDF2
from collections import defaultdict
import chromadb
from langsmith import traceable
from sentence_transformers import SentenceTransformer
from .database import get_book, get_all_books
import ebooklib
from ebooklib import epub
import mobi
import html2text

# Global embedding model instance
_model = None

def get_model():
    """Get or load the embedding model (singleton pattern)."""
    global _model
    if _model is None:
        print("Loading intfloat/multilingual-e5-small...")
        _model = SentenceTransformer("intfloat/multilingual-e5-small")
        print(f"✓ Model loaded! Dim: {_model.get_sentence_embedding_dimension()}, Max: {_model.max_seq_length}")
    return _model

@traceable(name="Generate Document Embedding", run_type="embedding")
def generate_embedding(text: str):
    """Generate embedding for documents with 'passage:' prefix."""
    model = get_model()
    emb = model.encode(f"passage: {text}", convert_to_tensor=True, normalize_embeddings=True)
    return emb.cpu().numpy()

@traceable(name="Generate Query Embedding", run_type="embedding")
def generate_query_embedding(text: str):
    """Generate embedding for queries with 'query:' prefix."""
    model = get_model()
    emb = model.encode(f"query: {text}", convert_to_tensor=True, normalize_embeddings=True)
    return emb.cpu().numpy()

# Configuration
CHUNK_SIZE_WORDS = 300
NUM_RANDOM_CHUNKS = 20
METADATA_WORDS = 150
TOP_K_INITIAL = 30
TOP_K_BOOKS = 2

# Global ChromaDB client and collection
_client = None
_collection = None

def get_collection(reset=False):
    """Get or create ChromaDB collection."""
    global _client, _collection
    
    if _client is None:
        _client = chromadb.PersistentClient(path="./chroma_e5")
    
    if reset:
        try:
            _client.delete_collection(name="books")
        except:
            pass
        _collection = None
    
    if _collection is None:
        _collection = _client.get_or_create_collection(
            name="books",
            metadata={"hnsw:space": "cosine"}
        )
        print("✓ ChromaDB ready.")
    
    return _collection

# Preprocessing functions
@traceable(name="Read PDF", run_type="tool")
def read_pdf(path: str) -> str:
    """Read and extract text from a PDF file."""
    text = ""
    try:
        with open(path, 'rb') as f:
            pdf = PyPDF2.PdfReader(f)
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
    except Exception as e:
        print(f"Error reading PDF {path}: {e}")
    return text

@traceable(name="Read EPUB", run_type="tool")
def read_epub(path: str) -> str:
    """Read and extract text from an EPUB file."""
    text = ""
    try:
        book = epub.read_epub(path)
        h = html2text.HTML2Text()
        h.ignore_links = True
        
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                content = item.get_content().decode('utf-8')
                text += h.handle(content) + "\n"
    except Exception as e:
        print(f"Error reading EPUB {path}: {e}")
    return text

@traceable(name="Read MOBI", run_type="tool")
def read_mobi(path: str) -> str:
    """Read and extract text from a MOBI file."""
    text = ""
    try:
        # mobi.extract() returns (extracted_dir, opf_file)
        import shutil
        extracted_dir, opf_file = mobi.extract(path)
        
        h = html2text.HTML2Text()
        h.ignore_links = True
        
        for root, dirs, files in os.walk(extracted_dir):
            for file in files:
                if file.endswith(('.html', '.htm', '.xhtml')):
                    with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                        text += h.handle(f.read()) + "\n"
        
        # Cleanup the extracted directory
        shutil.rmtree(extracted_dir)
    except Exception as e:
        print(f"Error reading MOBI {path}: {e}")
    return text

@traceable(name="Read Book", run_type="tool")
def read_book(path: str) -> str:
    """Read book content from file (supports TXT, PDF, EPUB, MOBI)."""
    if path.endswith('.pdf'):
        return read_pdf(path)
    elif path.endswith('.epub'):
        return read_epub(path)
    elif path.endswith('.mobi'):
        return read_mobi(path)
    
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

@traceable(name="Extract Book Metadata", run_type="tool")
def get_book_metadata(text: str) -> str:
    """Extract first N words as book summary."""
    return ' '.join(text.split()[:METADATA_WORDS])

@traceable(name="Create Random Chunks", run_type="tool")
def create_random_chunks(text: str):
    """Create random chunks from book text."""
    words = text.split()
    if len(words) < CHUNK_SIZE_WORDS:
        return [(' '.join(words), 0, len(words))]
    
    max_start = len(words) - CHUNK_SIZE_WORDS
    starts = sorted(random.sample(range(0, max_start), min(NUM_RANDOM_CHUNKS, max_start)))
    return [(' '.join(words[s:s+CHUNK_SIZE_WORDS]), s, s+CHUNK_SIZE_WORDS) for s in starts]

# Indexing
def index_books(force: bool = False):
    """Process and index only new books found in the database.
    
    Args:
        force: If True, reset the collection and re-index everything.
    """
    collection = get_collection(reset=force)
    
    # Get IDs already in vector store
    existing_ids = set()
    if not force:
        try:
            # We fetch metadatas to get the unique book_ids
            existing_data = collection.get(include=['metadatas'])
            if existing_data and existing_data['metadatas']:
                for meta in existing_data['metadatas']:
                    existing_ids.add(meta['book_id'])
            if existing_ids:
                print(f"ℹ️ Found {len(existing_ids)} books already indexed in ChromaDB.")
        except Exception as e:
            print(f"Note: Could not fetch existing IDs (likely empty collection): {e}")

    # Get all books from database
    db_books = get_all_books()
    print("Existing enbeds:", (existing_ids))
    if not db_books:
        print("No books found in database to index.")
        return
        
    all_chunks_data = []
    for book in db_books:
        book_id = book['book_id']
        book_name = book['book_name']
        file_path = book.get('file_path')
        
        # Skip if already indexed (unless forcing)
        print(f"Checking if book {book_id} is already indexed: {not force and book_id in existing_ids}")
        if not force and book_id in existing_ids:
            continue
            
        if not file_path or not os.path.exists(file_path):
            print(f"⚠️ Skipping {book_name}: File not found at {file_path}")
            continue
            
        print(f"📖 Indexing new book: {book_name} ({os.path.basename(file_path)})")
        text = read_book(file_path)
        metadata = get_book_metadata(text)
        
        for idx, (chunk, start, end) in enumerate(create_random_chunks(text)):
            all_chunks_data.append({
                'chunk_id': f"{book_id}_chunk_{idx:02d}",
                'book_id': book_id,
                'book_name': book_name,
                'chunk_idx': idx,
                'chunk_text': chunk,
                'combined_text': f"[BOOK]: {metadata}\n\n[EXCERPT]: {chunk}",
                'book_metadata': metadata,
                'word_start': start,
                'word_end': end
            })
    
    if not all_chunks_data:
        print("✓ All books are already up to date.")
        return
        
    print(f"\n✓ {len(all_chunks_data)} new chunks created.")
    
    # Generate embeddings and store
    print("Generating embeddings for new content...")
    for i in range(0, len(all_chunks_data), 10):
        batch = all_chunks_data[i:i+10]
        collection.add(
            ids=[c['chunk_id'] for c in batch],
            embeddings=[generate_embedding(c['combined_text']).tolist() for c in batch],
            documents=[c['chunk_text'] for c in batch],
            metadatas=[{
                'book_id': c['book_id'],
                'book_name': c['book_name'],
                'chunk_idx': c['chunk_idx'],
                'word_start': c['word_start'],
                'word_end': c['word_end'],
                'book_metadata': c['book_metadata'][:500]
            } for c in batch]
        )
        print(f"  {min(i+10, len(all_chunks_data))}/{len(all_chunks_data)}")
    
    print(f"\n✓ Indexing complete. Current store size: {collection.count()} chunks.")

# Retrieval
@traceable(name="ChromaDB Query", run_type="retriever")
def query_chromadb(query_embedding):
    """Query ChromaDB for similar chunks."""
    collection = get_collection()
    return collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=TOP_K_INITIAL,
        include=['documents', 'metadatas', 'distances']
    )

@traceable(name="Aggregate Book Scores", run_type="tool")
def aggregate_by_book(results):
    """Aggregate chunk scores by book ID."""
    book_scores = defaultdict(lambda: {'chunks': [], 'total': 0, 'best': 0})
    
    for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
        sim = 1 - dist
        book_scores[meta['book_id']]['chunks'].append({
            'doc': doc,
            'meta': meta,
            'sim': sim
        })
        book_scores[meta['book_id']]['total'] += sim
        book_scores[meta['book_id']]['best'] = max(book_scores[meta['book_id']]['best'], sim)
    
    return book_scores

@traceable(name="Rank Books", run_type="tool")
def rank_books(book_scores):
    """Rank books by combined score (best + average)."""
    ranked = []
    
    for bid, data in book_scores.items():
        score = 0.6 * data['best'] + 0.4 * (data['total'] / len(data['chunks']))
        
        # Get book metadata from database
        book_metadata = get_book(bid)
        
        ranked.append({
            'book_id': bid,
            'book_name': data['chunks'][0]['meta']['book_name'],
            'author_name': book_metadata['author_name'] if book_metadata else 'Unknown',
            'release_date': book_metadata['release_date'] if book_metadata else 'Unknown',
            'score': score,
            'chunk_count': len(data['chunks']),
            'best_chunk': data['chunks'][0]['doc'],
            'all_chunks': data['chunks']
        })
    
    ranked.sort(key=lambda x: x['score'], reverse=True)
    return ranked[:TOP_K_BOOKS]

@traceable(name="Retrieve and Aggregate", run_type="chain")
def retrieve_and_aggregate(query: str):
    """Full retrieval pipeline: embed query → search → aggregate → rank."""
    query_emb = generate_query_embedding(query)
    results = query_chromadb(query_emb)
    book_scores = aggregate_by_book(results)
    ranked_books = rank_books(book_scores)
    
    return {
        'query': query,
        'books': ranked_books
    }
