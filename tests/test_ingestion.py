from app.services.ingestion import DocumentIngestionService

service = DocumentIngestionService()
text = service.load_document("data/sample.txt")
chunks = service.chunk_text(text, "sample.txt")

print(f"Total chunks: {len(chunks)}")
print(chunks[0].content)
