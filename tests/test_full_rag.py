from pathlib import Path

from app.services.rag import RAGService
from app.services.ingestion import DocumentIngestionService


if __name__ == "__main__":
    print("Initializing RAG Service...")
    rag = RAGService()

    ingestion_service = DocumentIngestionService()

    data_folder = Path("data")

    all_chunks = []

    print("Loading documents from data folder...")
    print("Current working directory:", Path.cwd())
    print("Files inside data folder:")
    for f in data_folder.iterdir():
        print(" -", f.name)
    for file_path in data_folder.glob("*"):
        if file_path.suffix.lower() in [".pdf", ".txt"]:
            print(f"Processing: {file_path.name}")

            text = ingestion_service.load_document(str(file_path))
            chunks = ingestion_service.chunk_text(text, file_path.name)

            all_chunks.extend(chunks)

    if not all_chunks:
        raise ValueError("No valid documents found in data folder.")

    print(f"Total chunks created: {len(all_chunks)}")

    print("Indexing documents...")
    rag.index_documents(all_chunks)

    query = input("\nEnter your question: ")

    retrieved = rag.retrieve_context(query)

    print("\nRetrieved Chunks:\n")
    for r in retrieved:
        print("-----")
        print(r.content)

    answer = rag.answer_query(query)

    print("\nFinal Answer:\n")
    print(answer)
