from typing import List
import numpy as np
import faiss

from app.services.ingestion import DocumentChunk


class FAISSRetriever:
    """
    Handles vector storage and similarity search using FAISS.
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim

        # Using Inner Product because embeddings are normalized
        self.index = faiss.IndexFlatIP(embedding_dim)

        self.chunks: List[DocumentChunk] = []
        self.embeddings = None

    def add_documents(self, embeddings: np.ndarray, chunks: List[DocumentChunk]):
        """
        Add document embeddings and corresponding chunks to the index.
        """
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError("Embedding dimension mismatch.")

        self.index.add(embeddings)

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack((self.embeddings, embeddings))

        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 3) -> List[DocumentChunk]:
        """
        Search for top_k similar chunks.
        """
        if len(self.chunks) == 0:
            raise ValueError("No documents indexed.")

        query_embedding = np.expand_dims(query_embedding, axis=0)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[idx])

        return results
