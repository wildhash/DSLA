"""Tests for the StructuredMemory module."""

import os
import tempfile
import pytest
from datetime import datetime

from dsla.memory.structured_memory import StructuredMemory, MemoryEntry


class TestStructuredMemory:
    """Test StructuredMemory functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        yield path
        # Cleanup
        if os.path.exists(path):
            os.unlink(path)
    
    @pytest.fixture
    def memory(self, temp_db):
        """Create a StructuredMemory instance with temp database."""
        return StructuredMemory(db_path=temp_db)
    
    def test_initialization(self, memory):
        """Test memory initialization."""
        assert memory is not None
        assert os.path.exists(memory.db_path)
    
    def test_store_and_retrieve(self, memory):
        """Test storing and retrieving a memory entry."""
        entry = MemoryEntry(
            domain="test_domain",
            key="test_key",
            value={"data": "test_value"},
            metadata={"source": "test"}
        )
        
        # Store
        memory_id = memory.store(entry)
        assert memory_id is not None
        
        # Retrieve
        retrieved = memory.retrieve("test_domain", "test_key")
        assert retrieved is not None
        assert retrieved.domain == "test_domain"
        assert retrieved.key == "test_key"
        assert retrieved.value == {"data": "test_value"}
        assert retrieved.metadata == {"source": "test"}
    
    def test_retrieve_nonexistent(self, memory):
        """Test retrieving a non-existent entry."""
        result = memory.retrieve("nonexistent", "key")
        assert result is None
    
    def test_store_overwrites(self, memory):
        """Test that storing with same key overwrites."""
        entry1 = MemoryEntry(
            domain="test",
            key="key1",
            value={"version": 1}
        )
        entry2 = MemoryEntry(
            domain="test",
            key="key1",
            value={"version": 2}
        )
        
        memory.store(entry1)
        memory.store(entry2)
        
        retrieved = memory.retrieve("test", "key1")
        assert retrieved.value == {"version": 2}
    
    def test_query_by_domain(self, memory):
        """Test querying entries by domain."""
        # Store multiple entries
        for i in range(3):
            entry = MemoryEntry(
                domain="domain1",
                key=f"key{i}",
                value={"index": i}
            )
            memory.store(entry)
        
        entry = MemoryEntry(
            domain="domain2",
            key="key0",
            value={"index": 0}
        )
        memory.store(entry)
        
        # Query domain1
        results = memory.query(domain="domain1")
        assert len(results) == 3
        assert all(r.domain == "domain1" for r in results)
    
    def test_query_with_limit(self, memory):
        """Test querying with limit."""
        # Store multiple entries
        for i in range(10):
            entry = MemoryEntry(
                domain="test",
                key=f"key{i}",
                value={"index": i}
            )
            memory.store(entry)
        
        # Query with limit
        results = memory.query(domain="test", limit=5)
        assert len(results) == 5
    
    def test_query_with_offset(self, memory):
        """Test querying with offset."""
        # Store multiple entries
        for i in range(10):
            entry = MemoryEntry(
                domain="test",
                key=f"key{i}",
                value={"index": i}
            )
            memory.store(entry)
        
        # Query with offset
        results = memory.query(domain="test", limit=5, offset=5)
        assert len(results) == 5
    
    def test_delete(self, memory):
        """Test deleting an entry."""
        entry = MemoryEntry(
            domain="test",
            key="key1",
            value={"data": "value"}
        )
        memory.store(entry)
        
        # Delete
        deleted = memory.delete("test", "key1")
        assert deleted is True
        
        # Verify deleted
        retrieved = memory.retrieve("test", "key1")
        assert retrieved is None
    
    def test_delete_nonexistent(self, memory):
        """Test deleting a non-existent entry."""
        deleted = memory.delete("nonexistent", "key")
        assert deleted is False
    
    def test_clear_domain(self, memory):
        """Test clearing all entries in a domain."""
        # Store entries in multiple domains
        for i in range(5):
            entry = MemoryEntry(
                domain="domain1",
                key=f"key{i}",
                value={"index": i}
            )
            memory.store(entry)
        
        for i in range(3):
            entry = MemoryEntry(
                domain="domain2",
                key=f"key{i}",
                value={"index": i}
            )
            memory.store(entry)
        
        # Clear domain1
        count = memory.clear_domain("domain1")
        assert count == 5
        
        # Verify domain1 is empty
        results = memory.query(domain="domain1")
        assert len(results) == 0
        
        # Verify domain2 is intact
        results = memory.query(domain="domain2")
        assert len(results) == 3
    
    def test_memory_entry_with_timestamp(self, memory):
        """Test memory entry with custom timestamp."""
        now = datetime.now()
        entry = MemoryEntry(
            domain="test",
            key="key1",
            value={"data": "value"},
            timestamp=now
        )
        
        memory.store(entry)
        retrieved = memory.retrieve("test", "key1")
        
        assert retrieved is not None
        assert retrieved.timestamp is not None
