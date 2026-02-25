from typing import List

from app.services.embedding import EmbeddingService
from app.services.retriever import FAISSRetriever
from app.services.ingestion import DocumentChunk
from app.services.llm import LLMService


class RAGService:
    """
    Orchestrates the Retrieval-Augmented Generation pipeline.
    """

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.llm_service = LLMService()
        self.retriever = None

    def index_documents(self, chunks: List[DocumentChunk]):
        """
        Generate embeddings and index document chunks.
        """
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_service.embed_texts(texts)

        self.retriever = FAISSRetriever(embedding_dim=embeddings.shape[1])
        self.retriever.add_documents(embeddings, chunks)

    def retrieve_context(self, query: str, top_k: int = 3):
        if self.retriever is None:
            raise ValueError("Documents have not been indexed.")

        query_embedding = self.embedding_service.embed_query(query)
        results = self.retriever.search(query_embedding, top_k=top_k * 3)

        # Boost chunks containing "project"
        keyword = "project"
        boosted = []

        for chunk in results:
            if keyword.lower() in chunk.content.lower():
                boosted.insert(0, chunk)  # prioritize
            else:
                boosted.append(chunk)

        return boosted[:top_k]


    def build_prompt(self, query: str, retrieved_chunks: List[DocumentChunk]) -> str:
        """
        Construct a strictly grounded prompt with controlled length.
        """

        max_context_chars = 1200  # control size

        context_parts = []
        current_length = 0

        for chunk in retrieved_chunks:
            chunk_text = chunk.content

            if current_length + len(chunk_text) > max_context_chars:
                remaining = max_context_chars - current_length
                context_parts.append(chunk_text[:remaining])
                break
            else:
                context_parts.append(chunk_text)
                current_length += len(chunk_text)

        context = "\n\n".join(context_parts)

        prompt = f"""
    You are a document question answering assistant.

    Answer ONLY using the provided context.
    If the answer is not explicitly contained in the context, say:
    "I don't know."

    CONTEXT:
    {context}

    QUESTION:
    {query}

    ANSWER:
    """

        return prompt.strip()

    def answer_query(self, query: str, top_k: int = 1) -> str:
        """
        Full RAG pipeline:
        1. Retrieve relevant context
        2. Build grounded prompt
        3. Generate answer using LLM
        """
        if self.retriever is None:
            raise ValueError("Documents have not been indexed.")

        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.retrieve_context(query, top_k=top_k)

        # Step 2: Build prompt using retrieved context
        prompt = self.build_prompt(query, retrieved_chunks)

        # Step 3: Generate answer using LLM
        answer = self.llm_service.generate(prompt)

        return answer
