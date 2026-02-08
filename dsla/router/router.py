"""Router for selecting domain-specific adapters."""

from typing import Dict, List, Optional, Type
from dsla.adapters.base import Adapter


class Router:
    """
    Router for selecting the appropriate adapter based on domain.
    
    The router maintains a registry of available adapters and can
    intelligently select the best adapter for a given domain or task.
    """
    
    def __init__(self):
        """Initialize the router."""
        self._adapters: Dict[str, Adapter] = {}
        self._domain_keywords: Dict[str, List[str]] = {}
    
    def register_adapter(
        self,
        adapter: Adapter,
        keywords: Optional[List[str]] = None
    ) -> None:
        """
        Register an adapter with the router.
        
        Args:
            adapter: The adapter instance to register
            keywords: Optional list of keywords associated with this domain
        """
        domain = adapter.domain
        self._adapters[domain] = adapter
        
        if keywords:
            self._domain_keywords[domain] = keywords
        else:
            # Use domain name as default keyword
            self._domain_keywords[domain] = [domain.lower()]
    
    def get_adapter(self, domain: str) -> Optional[Adapter]:
        """
        Get an adapter by exact domain match.
        
        Args:
            domain: The domain identifier
            
        Returns:
            Adapter instance if found, None otherwise
        """
        return self._adapters.get(domain)
    
    def route(self, query: str) -> Optional[Adapter]:
        """
        Route a query to the most appropriate adapter.
        
        Args:
            query: The query or task description
            
        Returns:
            Best matching adapter or None
        """
        query_lower = query.lower()
        
        # Score each adapter based on keyword matches
        scores: Dict[str, int] = {}
        
        for domain, keywords in self._domain_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    score += 1
            
            if score > 0:
                scores[domain] = score
        
        # Return adapter with highest score
        if scores:
            best_domain = max(scores, key=scores.get)
            return self._adapters.get(best_domain)
        
        return None
    
    def list_adapters(self) -> List[str]:
        """
        List all registered adapter domains.
        
        Returns:
            List of domain names
        """
        return list(self._adapters.keys())
    
    def unregister_adapter(self, domain: str) -> bool:
        """
        Unregister an adapter.
        
        Args:
            domain: The domain to unregister
            
        Returns:
            True if unregistered, False if not found
        """
        if domain in self._adapters:
            del self._adapters[domain]
            if domain in self._domain_keywords:
                del self._domain_keywords[domain]
            return True
        return False
    
    def get_adapter_info(self, domain: str) -> Optional[Dict]:
        """
        Get information about a registered adapter.
        
        Args:
            domain: The domain to get info for
            
        Returns:
            Dictionary with adapter information or None
        """
        adapter = self._adapters.get(domain)
        if not adapter:
            return None
        
        return {
            "domain": domain,
            "keywords": self._domain_keywords.get(domain, []),
            "tools": [tool.name for tool in adapter.get_tools()],
            "schema": {
                "input": adapter.get_schema().input_schema,
                "output": adapter.get_schema().output_schema
            }
        }
