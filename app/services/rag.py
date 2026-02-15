from typing import List

from app.services.embedding import EmbeddingService
from app.services.retriever import FAISSRetriever
from app.services.ingestion import DocumentChunk


class RAGService:
    """
    Orchestrates the Retrieval-Augmented Generation pipeline.
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.retriever = None

    def index_documents(self, chunks: List[DocumentChunk]):
        """
        Generate embeddings and index document chunks.
        """
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(texts)

        self.retriever = FAISSRetriever(embedding_dim=embeddings.shape[1])
        self.retriever.add_documents(embeddings, chunks)

    def retrieve_context(self, query: str, top_k: int = 3) -> List[DocumentChunk]:
        """
        Retrieve relevant chunks for a given query.
        """
        if self.retriever is None:
            raise ValueError("Documents have not been indexed.")

        query_embedding = self.embedding_service.embed_query(query)
        results = self.retriever.search(query_embedding, top_k=top_k)

        return results

    def build_prompt(self, query: str, retrieved_chunks: List[DocumentChunk]) -> str:
        """
        Construct a grounded prompt for the LLM.
        """
        context = "\n\n".join([chunk.content for chunk in retrieved_chunks])

        prompt = f"""
Use the following context to answer the question.
If the answer is not contained in the context, say "I don't know."

Context:
{context}

Question:
{query}

Answer:
"""
        return prompt.strip()
