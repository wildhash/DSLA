"""FastAPI server for DSLA - Domain-Specific LLM Adapter."""

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from dsla.adapters.legal_doc import LegalDocAdapter
from dsla.adapters.trading_ops import TradingOpsAdapter
from dsla.memory.structured_memory import StructuredMemory, MemoryEntry
from dsla.rag.rag_module import RAGModule
from dsla.router.router import Router

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="DSLA - Domain-Specific LLM Adapter",
    description="A modular framework for adapting LLMs to specialized domains",
    version="0.1.0"
)

# Initialize components
router = Router()
memory = StructuredMemory(db_path=os.getenv("MEMORY_PATH", "./data/memory.db"))
rag = None  # Initialize on first use to avoid loading models at startup


def get_rag() -> RAGModule:
    """Lazy initialization of RAG module."""
    global rag
    if rag is None:
        use_local_embeddings = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
        use_faiss = os.getenv("USE_FAISS", "true").lower() == "true"

        try:
            local_embedding_dim = int(os.getenv("LOCAL_EMBEDDING_DIM", "384"))
        except ValueError:
            logger.exception("Invalid LOCAL_EMBEDDING_DIM; must be an integer.")
            raise HTTPException(status_code=500, detail="Invalid LOCAL_EMBEDDING_DIM; must be an integer.")
        try:
            rag = RAGModule(
                model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
                index_path=os.getenv("FAISS_INDEX_PATH", "./data/faiss_index"),
                use_faiss=use_faiss,
                use_local_embeddings=use_local_embeddings,
                local_embedding_dim=local_embedding_dim,
            )
        except RuntimeError as e:
            logger.exception("RAG initialization failed due to configuration")
            raise HTTPException(status_code=500, detail=f"RAG configuration error: {e}")
        except (ImportError, OSError):
            logger.exception("RAG initialization failed")
            raise HTTPException(
                status_code=500,
                detail="RAG initialization failed due to server configuration. Check server logs.",
            )
        except Exception:
            logger.exception("RAG initialization failed due to an unexpected server error")
            raise HTTPException(
                status_code=500,
                detail="RAG initialization failed due to an unexpected server error. Check server logs.",
            )
    return rag


# Register adapters
legal_adapter = LegalDocAdapter()
trading_adapter = TradingOpsAdapter()

router.register_adapter(
    legal_adapter,
    keywords=["legal", "contract", "agreement", "clause", "compliance", "law", "document"]
)
router.register_adapter(
    trading_adapter,
    keywords=["trading", "trade", "market", "stock", "crypto", "portfolio", "finance", "investment"]
)


# Request/Response models
class AdaptRequest(BaseModel):
    """Request model for /adapt endpoint."""
    
    domain: Optional[str] = Field(None, description="Specific domain to use")
    query: Optional[str] = Field(None, description="Query to route to appropriate domain")
    input_data: Dict[str, Any] = Field(..., description="Input data to adapt")


class AdaptResponse(BaseModel):
    """Response model for /adapt endpoint."""
    
    domain: str = Field(..., description="Domain that was used")
    adapted_input: Dict[str, Any] = Field(..., description="Adapted input data")
    prompt: str = Field(..., description="Generated prompt")
    tools: List[Dict[str, Any]] = Field(..., description="Available tools")
    schema: Dict[str, Any] = Field(..., description="Input/output schema")


class RunRequest(BaseModel):
    """Request model for /run endpoint."""
    
    domain: Optional[str] = Field(None, description="Specific domain to use")
    query: Optional[str] = Field(None, description="Query to route to appropriate domain")
    input_data: Dict[str, Any] = Field(..., description="Input data to process")
    use_rag: bool = Field(False, description="Whether to use RAG for context retrieval")
    rag_query: Optional[str] = Field(None, description="Query for RAG retrieval")
    save_to_memory: bool = Field(False, description="Whether to save results to memory")


class RunResponse(BaseModel):
    """Response model for /run endpoint."""
    
    domain: str = Field(..., description="Domain that was used")
    adapted_input: Dict[str, Any] = Field(..., description="Adapted input data")
    prompt: str = Field(..., description="Generated prompt")
    adapted_output: Dict[str, Any] = Field(..., description="Adapted output data")
    rag_context: Optional[List[Dict[str, Any]]] = Field(None, description="RAG retrieved context")
    memory_id: Optional[str] = Field(None, description="Memory entry ID if saved")


class MemoryStoreRequest(BaseModel):
    """Request model for storing memory."""
    
    domain: str = Field(..., description="Domain for this memory")
    key: str = Field(..., description="Unique key for this memory")
    value: Dict[str, Any] = Field(..., description="Memory content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RAGAddRequest(BaseModel):
    """Request model for adding documents to RAG."""
    
    documents: List[str] = Field(..., description="Documents to add")
    metadata: Optional[List[Dict[str, Any]]] = Field(None, description="Metadata for each document")


class RAGSearchRequest(BaseModel):
    """Request model for RAG search."""
    
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, description="Number of results to return")


# Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "DSLA - Domain-Specific LLM Adapter",
        "version": "0.1.0",
        "endpoints": {
            "adapt": "/adapt - Adapt input and get prompt/tools for a domain",
            "run": "/run - Full workflow execution with adaptation",
            "adapters": "/adapters - List available adapters",
            "memory": "/memory/* - Memory operations",
            "rag": "/rag/* - RAG operations"
        }
    }


@app.post("/adapt", response_model=AdaptResponse)
async def adapt(request: AdaptRequest):
    """
    Adapt input data for a specific domain.
    
    Returns the adapted input, generated prompt, available tools, and schema.
    """
    # Get adapter
    if request.domain:
        adapter = router.get_adapter(request.domain)
        if not adapter:
            raise HTTPException(status_code=404, detail=f"Domain '{request.domain}' not found")
    elif request.query:
        adapter = router.route(request.query)
        if not adapter:
            raise HTTPException(status_code=404, detail="No suitable adapter found for query")
    else:
        raise HTTPException(status_code=400, detail="Either 'domain' or 'query' must be provided")
    
    # Validate and adapt input
    try:
        adapter.validate_input(request.input_data)
        adapted_input = adapter.adapt_input(request.input_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Generate prompt
    prompt = adapter.format_prompt(**adapted_input)
    
    # Get tools and schema
    tools = [tool.model_dump() for tool in adapter.get_tools()]
    schema = adapter.get_schema().model_dump()
    
    return AdaptResponse(
        domain=adapter.domain,
        adapted_input=adapted_input,
        prompt=prompt,
        tools=tools,
        schema=schema
    )


@app.post("/run", response_model=RunResponse)
async def run(request: RunRequest):
    """
    Execute a full workflow with domain adaptation.
    
    This endpoint:
    1. Routes to appropriate adapter
    2. Adapts input
    3. Optionally retrieves RAG context
    4. Generates prompt
    5. Simulates output (in production, would call LLM)
    6. Adapts output
    7. Optionally saves to memory
    """
    # Get adapter
    if request.domain:
        adapter = router.get_adapter(request.domain)
        if not adapter:
            raise HTTPException(status_code=404, detail=f"Domain '{request.domain}' not found")
    elif request.query:
        adapter = router.route(request.query)
        if not adapter:
            raise HTTPException(status_code=404, detail="No suitable adapter found for query")
    else:
        raise HTTPException(status_code=400, detail="Either 'domain' or 'query' must be provided")
    
    # Validate and adapt input
    try:
        adapter.validate_input(request.input_data)
        adapted_input = adapter.adapt_input(request.input_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Generate prompt
    prompt = adapter.format_prompt(**adapted_input)
    
    # RAG retrieval if requested
    rag_context = None
    if request.use_rag and request.rag_query:
        rag_module = get_rag()
        results = rag_module.search(request.rag_query, top_k=5)
        rag_context = [
            {"document": doc, "score": score, "metadata": meta}
            for doc, score, meta in results
        ]
    
    # Simulate LLM output (in production, this would call actual LLM)
    # For now, return a mock response based on the domain
    if adapter.domain == "legal_doc":
        raw_output = {
            "key_clauses": ["Clause 1: Example", "Clause 2: Example"],
            "risks": ["Risk 1", "Risk 2"],
            "compliance_notes": ["Compliance note 1"],
            "recommendations": ["Recommendation 1"],
            "summary": f"Legal document analysis completed for {adapted_input.get('document_type', 'unknown')}"
        }
    elif adapter.domain == "trading_ops":
        raw_output = {
            "trend": "neutral",
            "support_levels": [100.0, 95.0],
            "resistance_levels": [110.0, 115.0],
            "signals": [{"type": "buy", "strength": "weak", "description": "Example signal"}],
            "risk_score": 5.0,
            "recommendations": ["Recommendation 1"],
            "summary": f"Trading analysis completed for {adapted_input.get('asset', 'unknown')}"
        }
    else:
        raw_output = {"summary": "Analysis completed"}
    
    # Adapt output
    adapted_output = adapter.adapt_output(raw_output)
    
    # Save to memory if requested
    memory_id = None
    if request.save_to_memory:
        memory_key = f"{adapter.domain}_{request.input_data.get('asset') or request.input_data.get('document_type', 'unknown')}"
        entry = MemoryEntry(
            domain=adapter.domain,
            key=memory_key,
            value=adapted_output,
            metadata={"input": adapted_input}
        )
        memory_id = memory.store(entry)
    
    return RunResponse(
        domain=adapter.domain,
        adapted_input=adapted_input,
        prompt=prompt,
        adapted_output=adapted_output,
        rag_context=rag_context,
        memory_id=memory_id
    )


@app.get("/adapters")
async def list_adapters():
    """List all available adapters."""
    adapters_list = []
    for domain in router.list_adapters():
        info = router.get_adapter_info(domain)
        if info:
            adapters_list.append(info)
    
    return {"adapters": adapters_list}


@app.post("/memory/store")
async def store_memory(request: MemoryStoreRequest):
    """Store a memory entry."""
    entry = MemoryEntry(
        domain=request.domain,
        key=request.key,
        value=request.value,
        metadata=request.metadata
    )
    memory_id = memory.store(entry)
    return {"id": memory_id, "status": "stored"}


@app.get("/memory/{domain}/{key}")
async def retrieve_memory(domain: str, key: str):
    """Retrieve a memory entry."""
    entry = memory.retrieve(domain, key)
    if not entry:
        raise HTTPException(status_code=404, detail="Memory entry not found")
    return entry.model_dump()


@app.get("/memory/{domain}")
async def query_memory(domain: str, limit: int = 100, offset: int = 0):
    """Query memory entries for a domain."""
    entries = memory.query(domain=domain, limit=limit, offset=offset)
    return {"entries": [entry.model_dump() for entry in entries]}


@app.post("/rag/add")
async def add_documents(request: RAGAddRequest):
    """Add documents to RAG index."""
    rag_module = get_rag()
    rag_module.add_documents(request.documents, request.metadata)
    return {"status": "added", "count": len(request.documents)}


@app.post("/rag/search")
async def search_documents(request: RAGSearchRequest):
    """Search for similar documents."""
    rag_module = get_rag()
    results = rag_module.search(request.query, top_k=request.top_k)
    
    return {
        "results": [
            {"document": doc, "score": float(score), "metadata": meta}
            for doc, score, meta in results
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    uvicorn.run(
        "dsla.server:app",
        host=host,
        port=port,
        reload=debug
    )
