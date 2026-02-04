"""RAG (Retrieval-Augmented Generation) module for DSLA."""

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


logger = logging.getLogger(__name__)


class LocalEmbeddingModel:
    """Deterministic local embedder for offline use.

    This is intended as a lightweight fallback for development/testing and
    environments where a sentence-transformers model is unavailable.

    It does not provide true semantic similarity and should not be used where
    embedding quality matters (for example, production RAG/search).
    """

    def __init__(self, embedding_dim: int = 384):
        self._embedding_dim = embedding_dim

    def get_sentence_embedding_dimension(self) -> int:
        return self._embedding_dim

    def encode(self, texts, show_progress_bar: bool = False, **_: Any):
        if isinstance(texts, str):
            texts = [texts]

        embeddings = np.zeros((len(texts), self._embedding_dim), dtype=np.float32)

        for i, text in enumerate(texts):
            tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
            if not tokens:
                continue

            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest[0:4], "little") % self._embedding_dim
                sign = 1.0 if (digest[4] & 1) == 0 else -1.0
                embeddings[i, idx] += sign

            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm

        return embeddings


class RAGModule:
    """
    Retrieval-Augmented Generation module.
    
    Provides embedding generation and similarity search capabilities
    using sentence-transformers and FAISS (with local fallback).
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: Optional[str] = None,
        use_faiss: bool = True,
        use_local_embeddings: bool = False,
        local_embedding_dim: int = 384,
    ):
        """
        Initialize the RAG module.
        
        Args:
            model_name: Name of the sentence transformer model
            index_path: Path to save/load FAISS index
            use_faiss: Whether to use FAISS for indexing (falls back to numpy if False)
            use_local_embeddings: Use a deterministic local embedder instead of sentence-transformers
            local_embedding_dim: Dimensionality for the local embedder
        """
        self.model_name = model_name
        self.index_path = index_path or "./data/faiss_index"
        self.use_faiss_requested = use_faiss
        self.faiss_available = FAISS_AVAILABLE
        self.use_faiss = self.use_faiss_requested and self.faiss_available
        self.use_local_embeddings = use_local_embeddings

        if self.use_faiss_requested and not self.faiss_available:
            logger.warning("FAISS requested but not available; falling back to numpy index.")
        
        # Initialize embedding model
        if use_local_embeddings:
            logger.warning(
                "use_local_embeddings=True: using LocalEmbeddingModel(dim=%d). This is a deterministic, "
                "non-semantic fallback intended for development/testing, not production-quality RAG/search.",
                local_embedding_dim,
            )
            self.model = LocalEmbeddingModel(embedding_dim=local_embedding_dim)
        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer(model_name)
        else:
            raise ImportError(
                "sentence-transformers is required unless use_local_embeddings=True. "
                "Install it with: pip install sentence-transformers (or set USE_LOCAL_EMBEDDINGS=true)"
            )

        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize index
        self.index = None
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        
        if self.use_faiss:
            self._init_faiss_index()
        else:
            self.embeddings: List[np.ndarray] = []
    
    def _init_faiss_index(self):
        """Initialize or load FAISS index."""
        if not FAISS_AVAILABLE:
            self.use_faiss = False
            self.embeddings = []
            return
        
        index_file = f"{self.index_path}.index"
        
        if os.path.exists(index_file):
            # Load existing index
            self.index = faiss.read_index(index_file)
        else:
            # Create new index
            self.index = faiss.IndexFlatL2(self.embedding_dim)

        if self.index is not None and self.index.d != self.embedding_dim:
            raise RuntimeError(
                f"FAISS index dimension {self.index.d} does not match embedding_dim {self.embedding_dim} "
                f"for index at '{self.index_path}.index'. If you changed LOCAL_EMBEDDING_DIM or "
                "EMBEDDING_MODEL, delete or rebuild this index file."
            )
    
    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Add documents to the index.
        
        Args:
            documents: List of document strings to index
            metadata: Optional metadata for each document
        """
        if not documents:
            return
        
        # Generate embeddings
        embeddings = self.model.encode(documents, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')
        
        # Store documents and metadata
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in documents])
        
        # Add to index
        if self.use_faiss:
            self.index.add(embeddings)
        else:
            self.embeddings.extend(embeddings)
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar documents.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of tuples (document, score, metadata)
        """
        if not self.documents:
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding).astype('float32')
        
        if self.use_faiss:
            # FAISS search
            distances, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
            
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < len(self.documents):
                    results.append((
                        self.documents[idx],
                        float(dist),
                        self.metadata[idx] if idx < len(self.metadata) else {}
                    ))
            return results
        else:
            # Numpy fallback - compute similarities manually
            embeddings_array = np.array(self.embeddings)
            
            # Compute L2 distances
            distances = np.linalg.norm(
                embeddings_array - query_embedding[0],
                axis=1
            )
            
            # Get top k indices
            top_indices = np.argsort(distances)[:top_k]
            
            results = []
            for idx in top_indices:
                results.append((
                    self.documents[idx],
                    float(distances[idx]),
                    self.metadata[idx] if idx < len(self.metadata) else {}
                ))
            return results
    
    def save_index(self) -> None:
        """Save the FAISS index to disk."""
        if self.use_faiss and self.index is not None:
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, f"{self.index_path}.index")
    
    def clear(self) -> None:
        """Clear all documents and reset the index."""
        self.documents = []
        self.metadata = []
        
        if self.use_faiss:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            self.embeddings = []
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of embeddings
        """
        return self.model.encode([text], show_progress_bar=False)[0]
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        return self.model.encode(texts, show_progress_bar=False)
