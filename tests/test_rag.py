from app.services.rag import RAGService
from app.services.ingestion import DocumentChunk

if __name__ == "__main__":
    rag = RAGService()

    texts = [
        "Artificial intelligence is transforming industries.",
        "Machine learning enables predictive analytics.",
        "The capital of France is Paris."
    ]

    chunks = [
        DocumentChunk(content=texts[i], metadata={"id": i})
        for i in range(len(texts))
    ]

    rag.index_documents(chunks)

    query = "What is the capital of France?"
    retrieved = rag.retrieve_context(query, top_k=2)

    print("Retrieved Context:")
    for r in retrieved:
        print("-", r.content)

    prompt = rag.build_prompt(query, retrieved)

    print("\nConstructed Prompt:\n")
    print(prompt)
