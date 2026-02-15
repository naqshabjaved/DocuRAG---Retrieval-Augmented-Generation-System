from app.services.embedding import EmbeddingService

if __name__ == "__main__":
    service = EmbeddingService()

    texts = [
        "Artificial intelligence is transforming industries.",
        "Machine learning enables predictive analytics."
    ]

    embeddings = service.embed_texts(texts)

    print("Embedding shape:", embeddings.shape)
    print("First vector sample:", embeddings[0][:5])
