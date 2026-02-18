"""Memory layer for user persona management.

Session history and user personas are both stored in the SDK's session database 
(memory/sessions/{user_id}.db) for consistency.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
from langsmith import traceable
from agents import function_tool

# Unified database for all user sessions and personas
SESSIONS_DB = Path("memory/user_sessions.db")
SESSIONS_DB.parent.mkdir(parents=True, exist_ok=True)

def init_persona_tables():
    """Ensure persona-related tables exist in the unified session database."""
    conn = sqlite3.connect(str(SESSIONS_DB))
    cursor = conn.cursor()
    
    # Create user personas table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_personas (
            user_id TEXT PRIMARY KEY,
            interests TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create persona updates table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS persona_updates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            update_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES user_personas (user_id) ON DELETE CASCADE
        )
    ''')
    
    # Create index for faster persona lookups
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_persona_updates_user_id 
        ON persona_updates (user_id, created_at DESC)
    ''')
    
    conn.commit()
    conn.close()

def get_persona_from_db(user_id: str) -> Optional[Dict]:
    """Get user persona from the unified session database."""
    if not SESSIONS_DB.exists():
        return None
        
    init_persona_tables()
    
    conn = sqlite3.connect(str(SESSIONS_DB))
    cursor = conn.cursor()
    
    try:
        # Get interests
        cursor.execute('SELECT interests FROM user_personas WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        interests = row[0].split(',') if row[0] else []
        
        # Get recent persona updates (last 5)
        cursor.execute('''
            SELECT update_text, created_at 
            FROM persona_updates 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT 5
        ''', (user_id,))
        
        updates = cursor.fetchall()
        return {
            'user_id': user_id,
            'interests': interests,
            'persona_updates': [
                {'update': update, 'timestamp': timestamp}
                for update, timestamp in updates
            ]
        }
    finally:
        conn.close()

def save_persona_update_to_db(user_id: str, update_text: str, new_interests: List[str] = None):
    """Save a persona update to the unified session database."""
    init_persona_tables()
    
    conn = sqlite3.connect(str(SESSIONS_DB))
    cursor = conn.cursor()
    
    try:
        # Ensure user exists in personas table
        cursor.execute('SELECT interests FROM user_personas WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()
        
        if row:
            # User exists, update interests if provided
            if new_interests:
                existing_interests = set(row[0].split(',')) if row[0] else set()
                existing_interests.update(new_interests)
                interests_str = ','.join(filter(None, existing_interests))
                
                cursor.execute('''
                    UPDATE user_personas 
                    SET interests = ?, updated_at = CURRENT_TIMESTAMP 
                    WHERE user_id = ?
                ''', (interests_str, user_id))
        else:
            # Create new user persona
            interests_str = ','.join(new_interests) if new_interests else ''
            cursor.execute('''
                INSERT INTO user_personas (user_id, interests) 
                VALUES (?, ?)
            ''', (user_id, interests_str))
        
        # Add persona update
        cursor.execute('''
            INSERT INTO persona_updates (user_id, update_text) 
            VALUES (?, ?)
        ''', (user_id, update_text))
        
        conn.commit()
        print(f"✓ Persona updated in unified session database for {user_id}: {update_text}")
    finally:
        conn.close()

@traceable(name="Get User Persona", run_type="tool")
def get_persona(user_id: str):
    """Retrieve user interests and persona from the session database."""
    persona_data = get_persona_from_db(user_id)
    
    if not persona_data:
        return "New user. No specific interests recorded yet."
    
    # Combine interests and persona updates
    interests = persona_data.get("interests", [])
    persona_updates = persona_data.get("persona_updates", [])
    
    result = []
    if interests:
        result.append(f"User is interested in: {', '.join(interests)}")
    
    if persona_updates:
        recent_updates = persona_updates[:5]
        updates_text = "; ".join([u["update"] for u in recent_updates])
        result.append(f"Recent preferences: {updates_text}")
    
    if not result:
        return "User has interacted but no specific interests identified yet."
    
    return ". ".join(result) + ". Keep these preferences in mind for recommendations."

@function_tool
def update_persona(user_id: str, update_text: str, interests: List[str] = None):
    """
    Update the user's persona with new information based on user chat and interactions.
    
    Args:
        user_id: Unique user identifier
        update_text: Description of new user preferences or interests
        interests: A list of key interest categories identified (e.g., ["Tech", "Philosophy"])
    """
    # Save to session database directly using interests provided by the agent
    save_persona_update_to_db(user_id, update_text, interests)
    
    return f"Persona updated successfully for {user_id}."

