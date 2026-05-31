"""
Vector store for semantic search using ChromaDB.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector database for storing and searching health content embeddings.
    
    Uses ChromaDB for local vector storage with multilingual embeddings.
    
    Example:
        store = VectorStore()
        store.add_documents([{"id": "1", "text": "...", "metadata": {...}}])
        results = store.search("vitamin D benefits", top_k=5)
    """
    
    DEFAULT_COLLECTION = "health_knowledge"
    
    def __init__(
        self,
        persist_dir: Path = None,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_dir: Directory for persistent storage
            embedding_model: Sentence transformer model for embeddings
        """
        self.persist_dir = Path(persist_dir) if persist_dir else Path("./data/chroma")
        self.embedding_model = embedding_model
        self._client = None
        self._embedding_fn = None
        self._collection = None
        
    def _get_client(self):
        """Get or create ChromaDB client."""
        if self._client is not None:
            return self._client
            
        import chromadb
        from chromadb.config import Settings
        
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        
        return self._client
    
    def _get_embedding_function(self):
        """Get embedding function."""
        if self._embedding_fn is not None:
            return self._embedding_fn
            
        from chromadb.utils import embedding_functions
        
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )
        
        return self._embedding_fn
    
    def get_collection(self, name: str = None):
        """Get or create a collection."""
        client = self._get_client()
        embedding_fn = self._get_embedding_function()
        
        collection_name = name or self.DEFAULT_COLLECTION
        
        return client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        collection_name: str = None,
    ) -> int:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of dicts with 'id', 'text', and optional 'metadata'
            collection_name: Target collection name
            
        Returns:
            Number of documents added
        """
        collection = self.get_collection(collection_name)
        
        ids = []
        texts = []
        metadatas = []
        
        for doc in documents:
            ids.append(doc["id"])
            texts.append(doc["text"])
            metadatas.append(doc.get("metadata", {}))
        
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )
        
        logger.info(f"Added {len(documents)} documents to {collection.name}")
        return len(documents)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        collection_name: str = None,
        filter_metadata: Dict = None,
    ) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            collection_name: Collection to search
            filter_metadata: Optional metadata filter
            
        Returns:
            List of results with 'id', 'text', 'metadata', 'score'
        """
        collection = self.get_collection(collection_name)
        
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"],
        )
        
        # Format results
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "score": 1 - results["distances"][0][i],  # Convert distance to similarity
            })
        
        return formatted
    
    def delete_documents(self, ids: List[str], collection_name: str = None):
        """Delete documents by ID."""
        collection = self.get_collection(collection_name)
        collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents")
    
    def get_stats(self, collection_name: str = None) -> Dict:
        """Get collection statistics."""
        collection = self.get_collection(collection_name)
        return {
            "name": collection.name,
            "count": collection.count(),
        }
