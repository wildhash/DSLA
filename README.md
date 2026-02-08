# DSLA - Domain-Specific LLM Adapter

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Domain-Specific LLM Adapter (DSLA) is a plug-and-play adaptation layer that sits on top of any base LLM. It's a modular framework for rapidly adapting large language models to specialized domains using lightweight adapters, structured memory, and retrieval-augmented reasoning—without retraining the base model.

## 🌟 Features

- **🔌 Modular Adapter System**: Plug-and-play adapters for different domains (legal, trading, etc.)
- **💾 Structured Memory**: Persistent SQLite-based memory for storing and retrieving domain-specific knowledge
- **🔍 RAG Module**: Retrieval-Augmented Generation with embeddings and FAISS/local fallback
- **🧭 Smart Router**: Intelligent routing to select the appropriate adapter based on query content
- **🚀 FastAPI Server**: RESTful API with `/adapt` and `/run` endpoints for easy integration
- **🛠️ Domain Tools**: Pre-built tools for legal document analysis and trading operations
- **📊 Type-Safe**: Built with Pydantic for robust data validation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       FastAPI Server                         │
│                    (/adapt, /run endpoints)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                          Router                              │
│              (Selects adapter by domain/query)              │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐      ┌──────────────┐
│   Legal Doc  │    │ Trading Ops  │      │   Custom     │
│   Adapter    │    │   Adapter    │      │   Adapters   │
└──────────────┘    └──────────────┘      └──────────────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐      ┌──────────────┐
│  Structured  │    │  RAG Module  │      │    Tools     │
│   Memory     │    │  (Embeddings)│      │  (Domain)    │
└──────────────┘    └──────────────┘      └──────────────┘
```

## 🚀 Quickstart

### Installation

```bash
# Clone the repository
git clone https://github.com/wildhash/DSLA.git
cd DSLA

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Configuration

Copy the example environment file and configure:

```bash
cp .env.example .env
```

Edit `.env` to configure your settings:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Memory Configuration
MEMORY_BACKEND=sqlite
MEMORY_PATH=./data/memory.db

# RAG Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
FAISS_INDEX_PATH=./data/faiss_index
# Best-effort FAISS index (set false to force numpy; falls back if FAISS is unavailable)
USE_FAISS=true
# Use deterministic local embeddings instead of sentence-transformers
USE_LOCAL_EMBEDDINGS=true
# Dimensionality for local embeddings (only used when USE_LOCAL_EMBEDDINGS=true; changing requires rebuilding index)
LOCAL_EMBEDDING_DIM=384

# Note: changing USE_LOCAL_EMBEDDINGS, LOCAL_EMBEDDING_DIM, or EMBEDDING_MODEL requires rebuilding/deleting the index
```

`USE_LOCAL_EMBEDDINGS=true` enables a deterministic, hashing-based embedder that works offline and is useful for local development/testing. It does not provide semantic similarity like sentence-transformers.

### Running the Server

```bash
# Start the FastAPI server
python -m dsla.server

# Or using uvicorn directly
uvicorn dsla.server:app --host 0.0.0.0 --port 8000 --reload
```

The server will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📖 Usage Examples

### Python API

#### Legal Document Analysis

```python
from dsla.adapters.legal_doc import LegalDocAdapter

# Initialize adapter
adapter = LegalDocAdapter()

# Prepare input
input_data = {
    "document_type": "contract",
    "content": "Your legal document text here...",
    "analysis_focus": "risk assessment"
}

# Adapt and validate
adapter.validate_input(input_data)
adapted_input = adapter.adapt_input(input_data)

# Generate prompt
prompt = adapter.format_prompt(**adapted_input)

# Get available tools
tools = adapter.get_tools()
print(f"Available tools: {[tool.name for tool in tools]}")
```

#### Trading Operations Analysis

```python
from dsla.adapters.trading_ops import TradingOpsAdapter

# Initialize adapter
adapter = TradingOpsAdapter()

# Prepare input
input_data = {
    "asset": "BTC/USD",
    "timeframe": "1d",
    "market_data": {
        "current_price": 45250.00,
        "volume": 12345678,
        "indicators": {"rsi_14": 62.5}
    },
    "analysis_type": "technical"
}

# Process
adapted_input = adapter.adapt_input(input_data)
prompt = adapter.format_prompt(**adapted_input)
```

#### Using the Router

```python
from dsla.router.router import Router
from dsla.adapters.legal_doc import LegalDocAdapter
from dsla.adapters.trading_ops import TradingOpsAdapter

# Initialize router
router = Router()

# Register adapters
router.register_adapter(
    LegalDocAdapter(),
    keywords=["legal", "contract", "law"]
)
router.register_adapter(
    TradingOpsAdapter(),
    keywords=["trading", "market", "stock"]
)

# Route by query
adapter = router.route("I need to analyze a legal contract")
print(f"Selected adapter: {adapter.domain}")
```

### REST API

#### Adapt Input for a Domain

```bash
curl -X POST "http://localhost:8000/adapt" \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "legal_doc",
    "input_data": {
      "document_type": "contract",
      "content": "Sample contract text"
    }
  }'
```

#### Run Full Workflow

```bash
curl -X POST "http://localhost:8000/run" \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "trading_ops",
    "input_data": {
      "asset": "BTC/USD",
      "market_data": {"price": 45000}
    },
    "save_to_memory": true
  }'
```

#### Query-Based Routing

```bash
curl -X POST "http://localhost:8000/adapt" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Analyze market trends for Bitcoin",
    "input_data": {
      "asset": "BTC/USD",
      "market_data": {"price": 45000}
    }
  }'
```

### Memory Operations

```bash
# Store memory
curl -X POST "http://localhost:8000/memory/store" \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "trading_ops",
    "key": "btc_analysis_2024",
    "value": {"trend": "bullish", "score": 8.5},
    "metadata": {"source": "api"}
  }'

# Retrieve memory
curl "http://localhost:8000/memory/trading_ops/btc_analysis_2024"

# Query domain memories
curl "http://localhost:8000/memory/trading_ops?limit=10"
```

### RAG Operations

```bash
# Add documents to RAG index
curl -X POST "http://localhost:8000/rag/add" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      "Machine learning is a subset of AI",
      "Deep learning uses neural networks"
    ]
  }'

# Search documents
curl -X POST "http://localhost:8000/rag/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence",
    "top_k": 5
  }'
```

## 🧪 Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=dsla --cov-report=html

# Run specific test file
pytest tests/test_adapters.py

# Run with verbose output
pytest -v
```

## 📦 Project Structure

```
DSLA/
├── dsla/
│   ├── __init__.py
│   ├── server.py              # FastAPI server
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── base.py            # Base Adapter class
│   │   ├── legal_doc.py       # Legal document adapter
│   │   └── trading_ops.py     # Trading operations adapter
│   ├── memory/
│   │   ├── __init__.py
│   │   └── structured_memory.py  # SQLite-based memory
│   ├── rag/
│   │   ├── __init__.py
│   │   └── rag_module.py      # RAG with embeddings
│   └── router/
│       ├── __init__.py
│       └── router.py          # Domain routing logic
├── tests/
│   ├── test_adapters.py
│   ├── test_memory.py
│   ├── test_rag.py
│   ├── test_router.py
│   └── test_server.py
├── examples/
│   ├── legal_doc_example.py
│   └── trading_ops_example.py
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 🔧 Creating Custom Adapters

Create your own domain-specific adapter by extending the base `Adapter` class:

```python
from dsla.adapters.base import Adapter, AdapterSchema, ToolDefinition
from typing import List, Dict, Any

class MyCustomAdapter(Adapter):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(domain="my_domain", config=config)
    
    def get_prompt_template(self) -> str:
        return """You are a {domain} expert. Task: {task}
        
        Context: {context}
        
        Please provide analysis..."""
    
    def get_schema(self) -> AdapterSchema:
        return AdapterSchema(
            input_schema={
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "context": {"type": "string"}
                },
                "required": ["task"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "result": {"type": "string"}
                },
                "required": ["result"]
            }
        )
    
    def get_tools(self) -> List[ToolDefinition]:
        return [
            ToolDefinition(
                name="my_tool",
                description="Does something useful",
                parameters={"type": "object"}
            )
        ]
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Embeddings powered by [Sentence Transformers](https://www.sbert.net/)
- Vector search with [FAISS](https://github.com/facebookresearch/faiss)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**DSLA** - Adapting LLMs to specialized domains, one adapter at a time. 🚀
