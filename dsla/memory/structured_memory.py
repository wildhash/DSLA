"""Structured Memory implementation for DSLA using SQLite."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MemoryEntry(BaseModel):
    """Represents a single memory entry."""
    
    id: Optional[str] = None
    domain: str = Field(..., description="Domain this memory belongs to")
    key: str = Field(..., description="Unique key for this memory")
    value: Dict[str, Any] = Field(..., description="The memory content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: Optional[datetime] = None


class StructuredMemory:
    """
    Structured memory storage using SQLite.
    
    Provides persistent storage for domain-specific memories with
    querying, retrieval, and update capabilities.
    """
    
    def __init__(self, db_path: str = "./data/memory.db"):
        """
        Initialize the structured memory.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    metadata TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(domain, key)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_domain ON memories(domain)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_key ON memories(key)
            """)
            conn.commit()
    
    def store(self, entry: MemoryEntry) -> str:
        """
        Store a memory entry.
        
        Args:
            entry: The memory entry to store
            
        Returns:
            The ID of the stored entry
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            value_json = json.dumps(entry.value)
            metadata_json = json.dumps(entry.metadata)
            
            cursor.execute("""
                INSERT OR REPLACE INTO memories (domain, key, value, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (
                entry.domain,
                entry.key,
                value_json,
                metadata_json,
                entry.timestamp or datetime.now()
            ))
            
            conn.commit()
            return str(cursor.lastrowid)
    
    def retrieve(self, domain: str, key: str) -> Optional[MemoryEntry]:
        """
        Retrieve a memory entry by domain and key.
        
        Args:
            domain: The domain to search in
            key: The key to retrieve
            
        Returns:
            MemoryEntry if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, domain, key, value, metadata, timestamp
                FROM memories
                WHERE domain = ? AND key = ?
            """, (domain, key))
            
            row = cursor.fetchone()
            if row:
                return MemoryEntry(
                    id=str(row[0]),
                    domain=row[1],
                    key=row[2],
                    value=json.loads(row[3]),
                    metadata=json.loads(row[4]) if row[4] else {},
                    timestamp=datetime.fromisoformat(row[5]) if row[5] else None
                )
            return None
    
    def query(
        self,
        domain: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[MemoryEntry]:
        """
        Query memory entries with optional filtering.
        
        Args:
            domain: Optional domain filter
            limit: Maximum number of entries to return
            offset: Number of entries to skip
            
        Returns:
            List of matching MemoryEntry objects
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            if domain:
                cursor.execute("""
                    SELECT id, domain, key, value, metadata, timestamp
                    FROM memories
                    WHERE domain = ?
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                """, (domain, limit, offset))
            else:
                cursor.execute("""
                    SELECT id, domain, key, value, metadata, timestamp
                    FROM memories
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
            
            entries = []
            for row in cursor.fetchall():
                entries.append(MemoryEntry(
                    id=str(row[0]),
                    domain=row[1],
                    key=row[2],
                    value=json.loads(row[3]),
                    metadata=json.loads(row[4]) if row[4] else {},
                    timestamp=datetime.fromisoformat(row[5]) if row[5] else None
                ))
            
            return entries
    
    def delete(self, domain: str, key: str) -> bool:
        """
        Delete a memory entry.
        
        Args:
            domain: The domain of the entry
            key: The key of the entry
            
        Returns:
            True if deleted, False if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM memories
                WHERE domain = ? AND key = ?
            """, (domain, key))
            conn.commit()
            return cursor.rowcount > 0
    
    def clear_domain(self, domain: str) -> int:
        """
        Clear all memories for a domain.
        
        Args:
            domain: The domain to clear
            
        Returns:
            Number of entries deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM memories WHERE domain = ?
            """, (domain,))
            conn.commit()
            return cursor.rowcount
