"""Tests for the Router module."""

import pytest
from dsla.router.router import Router
from dsla.adapters.legal_doc import LegalDocAdapter
from dsla.adapters.trading_ops import TradingOpsAdapter


class TestRouter:
    """Test Router functionality."""
    
    @pytest.fixture
    def router(self):
        """Create a router instance."""
        return Router()
    
    @pytest.fixture
    def router_with_adapters(self, router):
        """Create a router with registered adapters."""
        legal_adapter = LegalDocAdapter()
        trading_adapter = TradingOpsAdapter()
        
        router.register_adapter(
            legal_adapter,
            keywords=["legal", "contract", "law", "compliance"]
        )
        router.register_adapter(
            trading_adapter,
            keywords=["trading", "market", "stock", "crypto"]
        )
        
        return router
    
    def test_initialization(self, router):
        """Test router initialization."""
        assert router is not None
        assert len(router.list_adapters()) == 0
    
    def test_register_adapter(self, router):
        """Test registering an adapter."""
        adapter = LegalDocAdapter()
        router.register_adapter(adapter, keywords=["legal", "law"])
        
        assert "legal_doc" in router.list_adapters()
        assert router.get_adapter("legal_doc") is adapter
    
    def test_register_adapter_without_keywords(self, router):
        """Test registering an adapter without explicit keywords."""
        adapter = LegalDocAdapter()
        router.register_adapter(adapter)
        
        # Should use domain name as keyword
        assert "legal_doc" in router.list_adapters()
    
    def test_get_adapter(self, router_with_adapters):
        """Test getting an adapter by domain."""
        adapter = router_with_adapters.get_adapter("legal_doc")
        
        assert adapter is not None
        assert adapter.domain == "legal_doc"
    
    def test_get_nonexistent_adapter(self, router_with_adapters):
        """Test getting a non-existent adapter."""
        adapter = router_with_adapters.get_adapter("nonexistent")
        assert adapter is None
    
    def test_route_by_keyword(self, router_with_adapters):
        """Test routing based on keywords."""
        # Legal keywords
        adapter = router_with_adapters.route("I need to analyze a legal contract")
        assert adapter is not None
        assert adapter.domain == "legal_doc"
        
        # Trading keywords
        adapter = router_with_adapters.route("What is the market trend for this stock?")
        assert adapter is not None
        assert adapter.domain == "trading_ops"
    
    def test_route_no_match(self, router_with_adapters):
        """Test routing when no adapter matches."""
        adapter = router_with_adapters.route("Tell me about quantum physics")
        assert adapter is None
    
    def test_route_best_match(self, router_with_adapters):
        """Test that routing picks the best matching adapter."""
        # Query with multiple legal keywords should match legal adapter
        adapter = router_with_adapters.route(
            "Review this legal contract for compliance issues"
        )
        assert adapter is not None
        assert adapter.domain == "legal_doc"
    
    def test_list_adapters(self, router_with_adapters):
        """Test listing all adapters."""
        adapters = router_with_adapters.list_adapters()
        
        assert len(adapters) == 2
        assert "legal_doc" in adapters
        assert "trading_ops" in adapters
    
    def test_unregister_adapter(self, router_with_adapters):
        """Test unregistering an adapter."""
        result = router_with_adapters.unregister_adapter("legal_doc")
        
        assert result is True
        assert "legal_doc" not in router_with_adapters.list_adapters()
        assert router_with_adapters.get_adapter("legal_doc") is None
    
    def test_unregister_nonexistent_adapter(self, router_with_adapters):
        """Test unregistering a non-existent adapter."""
        result = router_with_adapters.unregister_adapter("nonexistent")
        assert result is False
    
    def test_get_adapter_info(self, router_with_adapters):
        """Test getting adapter information."""
        info = router_with_adapters.get_adapter_info("legal_doc")
        
        assert info is not None
        assert info["domain"] == "legal_doc"
        assert "keywords" in info
        assert "tools" in info
        assert "schema" in info
        assert len(info["keywords"]) > 0
    
    def test_get_adapter_info_nonexistent(self, router_with_adapters):
        """Test getting info for non-existent adapter."""
        info = router_with_adapters.get_adapter_info("nonexistent")
        assert info is None
    
    def test_multiple_keyword_matches(self, router_with_adapters):
        """Test routing with multiple keyword matches."""
        # Query with both legal and trading keywords
        # Should pick the one with more matches
        adapter = router_with_adapters.route(
            "Legal compliance for trading operations and market contracts"
        )
        
        # Both adapters could match, but one should be selected
        assert adapter is not None
        assert adapter.domain in ["legal_doc", "trading_ops"]
