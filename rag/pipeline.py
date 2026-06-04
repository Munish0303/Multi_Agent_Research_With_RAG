import os
from pathlib import Path
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import CHROMA_PERSIST_DIR, UPLOAD_DIR
from config.providers import get_embeddings, collection_suffix

_CHROMA_SETTINGS = ChromaSettings(anonymized_telemetry=False)


class RAGPipeline:
    def __init__(self, collection_name: str = "research_docs"):
        # Suffix keeps 384-dim (cloud) and 768-dim (local) vectors in separate
        # collections so switching providers never causes a dimension mismatch.
        self.collection_name = collection_name + collection_suffix()
        self.embeddings = get_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vectorstore: Optional[Chroma] = None
        self._init_vectorstore()

    def _init_vectorstore(self):
        # Create the client directly so telemetry is disabled at the source,
        # rather than relying on langchain_chroma to propagate client_settings.
        client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=_CHROMA_SETTINGS,
        )
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            client=client,
        )

    def _load_pdf(self, path: Path) -> List[Document]:
        """Load a PDF, tolerating malformed files. Tries the normal loader,
        then a lenient page-by-page pypdf read that skips unreadable pages."""
        try:
            return PyPDFLoader(str(path)).load()
        except Exception:
            pass  # fall through to the lenient reader

        from pypdf import PdfReader
        docs: List[Document] = []
        try:
            reader = PdfReader(str(path), strict=False)
            pages = reader.pages
        except Exception as exc:
            raise ValueError(
                f"This PDF could not be read ({type(exc).__name__}: {exc}). "
                "It may be truncated, corrupted, or not a valid PDF. Try "
                "re-saving/re-exporting it, or upload a .txt/.md version."
            )

        for i, page in enumerate(pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                continue  # skip pages pypdf can't decode
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": str(path), "page": i},
                ))
        if not docs:
            raise ValueError(
                "Could not extract any readable text from this PDF — the file "
                "may be scanned (image-only) or password-protected. A text "
                "(.txt/.md) version will work."
            )
        return docs

    def ingest_file(self, file_path: str) -> int:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix.lower() == ".pdf":
            documents = self._load_pdf(path)
        elif path.suffix.lower() in [".txt", ".md"]:
            documents = TextLoader(str(path), encoding="utf-8").load()
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        chunks = self.text_splitter.split_documents(documents)

        for chunk in chunks:
            chunk.metadata["source_file"] = path.name

        if not chunks:
            raise ValueError("No text chunks were produced from this file.")

        self.vectorstore.add_documents(chunks)
        return len(chunks)

    def ingest_directory(self, directory: str = None) -> int:
        directory = directory or UPLOAD_DIR
        total = 0
        for file in Path(directory).iterdir():
            if file.suffix.lower() in [".pdf", ".txt", ".md"]:
                try:
                    count = self.ingest_file(str(file))
                    total += count
                    print(f"  Ingested {file.name}: {count} chunks")
                except Exception as e:
                    print(f"  Error ingesting {file.name}: {e}")
        return total

    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        if self.vectorstore is None:
            return []
        return self.vectorstore.similarity_search(query, k=k)

    def get_retriever(self, k: int = 5):
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def collection_count(self) -> int:
        try:
            return self.vectorstore._collection.count()
        except Exception:
            return 0
