from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader


class DocumentChunk:
    """
    Represents a single chunk of text extracted from a document.
    """

    def __init__(self, content: str, metadata: Dict):
        self.content = content
        self.metadata = metadata


class DocumentIngestionService:
    """
    Handles document loading, cleaning, and chunking.
    """

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_document(self, file_path: str) -> str:
        """
        Loads text from a PDF or TXT file.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() == ".pdf":
            return self._load_pdf(path)

        elif path.suffix.lower() == ".txt":
            return path.read_text(encoding="utf-8")

        else:
            raise ValueError("Unsupported file type. Only PDF and TXT are supported.")

    def _load_pdf(self, path: Path) -> str:
        """
        Extract text from PDF file.
        """
        reader = PdfReader(str(path))
        text = ""

        for page_number, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n\n[Page {page_number + 1}]\n"
                text += page_text

        return text

    def chunk_text(self, text: str, source_name: str) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks.
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk_text = text[start:end]

            metadata = {
                "source": source_name,
                "start_index": start,
                "end_index": end
            }

            chunks.append(DocumentChunk(content=chunk_text, metadata=metadata))

            start += self.chunk_size - self.chunk_overlap

        return chunks
