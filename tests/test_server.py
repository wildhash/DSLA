"""Tests for the FastAPI server endpoints."""

import pytest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

from dsla.server import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestServerEndpoints:
    """Test FastAPI server endpoints."""
    
    def test_root_endpoint(self, client):
        """Test the root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data
    
    def test_list_adapters(self, client):
        """Test listing available adapters."""
        response = client.get("/adapters")
        
        assert response.status_code == 200
        data = response.json()
        assert "adapters" in data
        assert len(data["adapters"]) >= 2
        
        domains = [adapter["domain"] for adapter in data["adapters"]]
        assert "legal_doc" in domains
        assert "trading_ops" in domains
    
    def test_adapt_legal_doc(self, client):
        """Test /adapt endpoint with legal document."""
        request_data = {
            "domain": "legal_doc",
            "input_data": {
                "document_type": "contract",
                "content": "Sample contract text"
            }
        }
        
        response = client.post("/adapt", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["domain"] == "legal_doc"
        assert "adapted_input" in data
        assert "prompt" in data
        assert "tools" in data
        assert "schema" in data
        assert len(data["tools"]) > 0
    
    def test_adapt_trading_ops(self, client):
        """Test /adapt endpoint with trading operations."""
        request_data = {
            "domain": "trading_ops",
            "input_data": {
                "asset": "BTC/USD",
                "market_data": {"price": 45000}
            }
        }
        
        response = client.post("/adapt", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["domain"] == "trading_ops"
        assert "adapted_input" in data
        assert data["adapted_input"]["asset"] == "BTC/USD"
    
    def test_adapt_with_routing(self, client):
        """Test /adapt with query-based routing."""
        request_data = {
            "query": "I need to analyze a legal contract",
            "input_data": {
                "document_type": "contract",
                "content": "Sample text"
            }
        }
        
        response = client.post("/adapt", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["domain"] == "legal_doc"
    
    def test_adapt_invalid_domain(self, client):
        """Test /adapt with invalid domain."""
        request_data = {
            "domain": "nonexistent",
            "input_data": {"test": "data"}
        }
        
        response = client.post("/adapt", json=request_data)
        assert response.status_code == 404
    
    def test_adapt_missing_domain_and_query(self, client):
        """Test /adapt without domain or query."""
        request_data = {
            "input_data": {"test": "data"}
        }
        
        response = client.post("/adapt", json=request_data)
        assert response.status_code == 400
    
    def test_run_legal_doc(self, client):
        """Test /run endpoint with legal document."""
        request_data = {
            "domain": "legal_doc",
            "input_data": {
                "document_type": "contract",
                "content": "Sample contract"
            }
        }
        
        response = client.post("/run", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["domain"] == "legal_doc"
        assert "adapted_input" in data
        assert "adapted_output" in data
        assert "prompt" in data
        assert "summary" in data["adapted_output"]
    
    def test_run_trading_ops(self, client):
        """Test /run endpoint with trading operations."""
        request_data = {
            "domain": "trading_ops",
            "input_data": {
                "asset": "BTC/USD",
                "market_data": {"price": 45000}
            }
        }
        
        response = client.post("/run", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["domain"] == "trading_ops"
        assert "adapted_output" in data
        assert "trend" in data["adapted_output"]
    
    def test_run_with_memory(self, client):
        """Test /run with memory storage."""
        request_data = {
            "domain": "legal_doc",
            "input_data": {
                "document_type": "contract",
                "content": "Sample"
            },
            "save_to_memory": True
        }
        
        response = client.post("/run", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "memory_id" in data
        assert data["memory_id"] is not None
    
    def test_memory_store_and_retrieve(self, client):
        """Test storing and retrieving memory."""
        # Store
        store_data = {
            "domain": "test_domain",
            "key": "test_key",
            "value": {"data": "test_value"},
            "metadata": {"source": "test"}
        }
        
        response = client.post("/memory/store", json=store_data)
        assert response.status_code == 200
        
        # Retrieve
        response = client.get("/memory/test_domain/test_key")
        assert response.status_code == 200
        data = response.json()
        assert data["domain"] == "test_domain"
        assert data["key"] == "test_key"
        assert data["value"]["data"] == "test_value"
    
    def test_memory_query(self, client):
        """Test querying memory by domain."""
        # Store some entries
        for i in range(3):
            store_data = {
                "domain": "query_test",
                "key": f"key{i}",
                "value": {"index": i}
            }
            client.post("/memory/store", json=store_data)
        
        # Query
        response = client.get("/memory/query_test")
        assert response.status_code == 200
        data = response.json()
        assert "entries" in data
        assert len(data["entries"]) >= 3
    
    def test_rag_add_and_search(self, client):
        """Test adding documents and searching in RAG."""
        # Mock the sentence transformer to avoid network calls
        with patch('dsla.rag.rag_module.SentenceTransformer') as MockModel:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            import numpy as np
            mock_model.encode.side_effect = lambda texts, **kwargs: np.random.randn(
                len(texts) if isinstance(texts, list) else 1, 384
            ).astype('float32')
            MockModel.return_value = mock_model
            
            # Add documents
            add_data = {
                "documents": [
                    "Machine learning is a subset of AI",
                    "Deep learning uses neural networks"
                ]
            }
            
            response = client.post("/rag/add", json=add_data)
            assert response.status_code == 200
            
            # Search
            search_data = {
                "query": "artificial intelligence",
                "top_k": 2
            }
            
            response = client.post("/rag/search", json=search_data)
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
