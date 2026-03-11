from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import RAW_PDFS_DIR, VECTOR_STORE_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from src.ingest.translator import is_greek, translate_to_english


def load_documents():
    documents = []

    # Load PDFs recursively
    pdf_loader = PyPDFDirectoryLoader(str(RAW_PDFS_DIR), recursive=True)
    pdf_docs = pdf_loader.load()
    print(f"  PDFs: {len(pdf_docs)} pages loaded.")
    documents.extend(pdf_docs)

    # Load DOCX recursively
    docx_loader = DirectoryLoader(
        str(RAW_PDFS_DIR),
        glob="**/*.docx",
        loader_cls=Docx2txtLoader,
        recursive=True
    )
    docx_docs = docx_loader.load()
    print(f"  DOCX: {len(docx_docs)} documents loaded.")
    documents.extend(docx_docs)

    return documents


def build_index():
    print("Loading documents...")
    documents = load_documents()

    if not documents:
        print("No documents found.")
        return

    print(f"Total: {len(documents)} documents loaded.")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Generating embeddings...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)

    print("Saving vector store...")
    vector_store.save_local(str(VECTOR_STORE_DIR))
    print(f"Done. Index saved to {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    build_index()