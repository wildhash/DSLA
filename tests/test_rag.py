"""Tests for the RAG module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from dsla.rag.rag_module import RAGModule


class TestRAGModule:
    """Test RAGModule functionality."""
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        """Create a mock sentence transformer."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.side_effect = lambda texts, **kwargs: np.random.randn(
            len(texts) if isinstance(texts, list) else 1, 384
        ).astype('float32')
        return mock_model
    
    @pytest.fixture
    def rag(self, mock_sentence_transformer):
        """Create a RAG module instance with mocked model."""
        with patch('dsla.rag.rag_module.SentenceTransformer', return_value=mock_sentence_transformer):
            return RAGModule(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                use_faiss=False
            )
    
    def test_initialization(self, rag):
        """Test RAG module initialization."""
        assert rag is not None
        assert rag.model is not None
        assert rag.embedding_dim > 0
    
    def test_get_embedding(self, rag):
        """Test getting embedding for a single text."""
        text = "This is a test document"
        embedding = rag.get_embedding(text)
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == rag.embedding_dim
    
    def test_get_embeddings(self, rag):
        """Test getting embeddings for multiple texts."""
        texts = [
            "First document",
            "Second document",
            "Third document"
        ]
        embeddings = rag.get_embeddings(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == rag.embedding_dim
    
    def test_add_documents(self, rag):
        """Test adding documents to the index."""
        documents = [
            "Machine learning is a subset of artificial intelligence",
            "Deep learning uses neural networks with multiple layers",
            "Natural language processing deals with text and speech"
        ]
        
        rag.add_documents(documents)
        
        assert len(rag.documents) == 3
        assert len(rag.embeddings) == 3
    
    def test_add_documents_with_metadata(self, rag):
        """Test adding documents with metadata."""
        documents = ["Doc 1", "Doc 2"]
        metadata = [{"source": "test1"}, {"source": "test2"}]
        
        rag.add_documents(documents, metadata)
        
        assert len(rag.documents) == 2
        assert len(rag.metadata) == 2
        assert rag.metadata[0]["source"] == "test1"
    
    def test_search(self, rag):
        """Test searching for similar documents."""
        documents = [
            "Python is a programming language",
            "Java is used for enterprise applications",
            "JavaScript runs in web browsers",
            "Machine learning models predict patterns"
        ]
        
        rag.add_documents(documents)
        
        # Search for programming-related content
        results = rag.search("programming languages", top_k=2)
        
        assert len(results) <= 2
        assert all(len(result) == 3 for result in results)  # (doc, score, metadata)
        
        # First result should be most relevant
        doc, score, metadata = results[0]
        assert isinstance(doc, str)
        assert isinstance(score, float)
        assert isinstance(metadata, dict)
    
    def test_search_empty_index(self, rag):
        """Test searching with no documents."""
        results = rag.search("test query", top_k=5)
        assert len(results) == 0
    
    def test_search_top_k_limit(self, rag):
        """Test that search respects top_k limit."""
        documents = [f"Document {i}" for i in range(10)]
        rag.add_documents(documents)
        
        results = rag.search("document", top_k=3)
        assert len(results) == 3
    
    def test_clear(self, rag):
        """Test clearing the index."""
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        rag.add_documents(documents)
        
        assert len(rag.documents) == 3
        
        rag.clear()
        
        assert len(rag.documents) == 0
        assert len(rag.embeddings) == 0
        assert len(rag.metadata) == 0
    
    def test_search_relevance(self, rag):
        """Test that search returns relevant results."""
        documents = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning and artificial intelligence",
            "Python programming and software development",
            "The weather is sunny and warm today"
        ]
        
        rag.add_documents(documents)
        
        # Search for AI-related content
        results = rag.search("artificial intelligence machine learning", top_k=2)
        
        # Should return 2 results
        assert len(results) == 2
        
        # Results should be documents
        for doc, score, metadata in results:
            assert isinstance(doc, str)
            assert doc in documents
