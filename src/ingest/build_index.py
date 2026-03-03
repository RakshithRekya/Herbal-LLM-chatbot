from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import RAW_PDFS_DIR, VECTOR_STORE_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP


def load_documents():
    documents = []

    # Load PDFs
    pdf_loader = PyPDFDirectoryLoader(str(RAW_PDFS_DIR))
    pdf_docs = pdf_loader.load()
    print(f"  PDFs: {len(pdf_docs)} pages loaded.")
    documents.extend(pdf_docs)

    # Load DOCX files
    docx_loader = DirectoryLoader(
        str(RAW_PDFS_DIR),
        glob="**/*.docx",
        loader_cls=__import__(
            "langchain_community.document_loaders",
            fromlist=["Docx2txtLoader"]
        ).Docx2txtLoader
    )
    docx_docs = docx_loader.load()
    print(f"  DOCX: {len(docx_docs)} documents loaded.")
    documents.extend(docx_docs)

    return documents


def build_index():
    print("Loading documents...")
    documents = load_documents()

    if not documents:
        print("No documents found. Add PDFs or DOCX files to data/raw_pdfs/ and try again.")
        return

    print(f"Total: {len(documents)} documents loaded.")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Generating embeddings — this may take a few minutes on CPU...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)

    print("Saving vector store...")
    vector_store.save_local(str(VECTOR_STORE_DIR))
    print(f"Done. Index saved to {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    build_index()
