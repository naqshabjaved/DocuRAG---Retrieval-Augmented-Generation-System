from app.services.embedding import EmbeddingService
from app.services.ingestion import DocumentChunk
from app.services.retriever import FAISSRetriever

if __name__ == "__main__":
    embedding_service = EmbeddingService()

    texts = [
        "Artificial intelligence is transforming industries.",
        "Machine learning enables predictive analytics.",
        "The capital of France is Paris."
    ]

    embeddings = embedding_service.embed_texts(texts)

    chunks = [
        DocumentChunk(content=texts[i], metadata={"id": i})
        for i in range(len(texts))
    ]

    retriever = FAISSRetriever(embedding_dim=embeddings.shape[1])
    retriever.add_documents(embeddings, chunks)

    query = "What is the capital of France?"
    query_embedding = embedding_service.embed_query(query)

    results = retriever.search(query_embedding, top_k=2)

    print("Top Results:")
    for r in results:
        print("-", r.content)
