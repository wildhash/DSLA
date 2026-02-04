"""
DSLA - Domain-Specific LLM Adapter

A modular framework for adapting LLMs to specialized domains using
lightweight adapters, structured memory, and retrieval-augmented reasoning.
"""

__version__ = "0.1.0"

from dsla.adapters.base import Adapter
from dsla.memory.structured_memory import StructuredMemory
from dsla.rag.rag_module import RAGModule
from dsla.router.router import Router

__all__ = ["Adapter", "StructuredMemory", "RAGModule", "Router"]
