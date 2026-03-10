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

    pdf_loader = PyPDFDirectoryLoader(str(RAW_PDFS_DIR))
    pdf_docs = pdf_loader.load()
    print(f"  PDFs: {len(pdf_docs)} pages loaded.")
    documents.extend(pdf_docs)

    docx_loader = DirectoryLoader(
        str(RAW_PDFS_DIR),
        glob="**/*.docx",
        loader_cls=Docx2txtLoader
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

    # Translate Greek chunks and build bilingual index
    print("Building bilingual index...")
    all_chunks = []

    for i, chunk in enumerate(chunks):
        # Always keep original chunk
        all_chunks.append(chunk)

        # If Greek, translate and add English version alongside
        if is_greek(chunk.page_content):
            print(f"  Translating chunk {i+1}/{len(chunks)}...")
            translated = translate_to_english(chunk.page_content)
            if translated:
                all_chunks.append(Document(
                    page_content=translated,
                    metadata={**chunk.metadata, "language": "en", "translated": True}
                ))

    greek_count = sum(1 for c in all_chunks if c.metadata.get("translated"))
    print(f"  Translated {greek_count} Greek chunks to English.")
    print(f"  Total chunks in index: {len(all_chunks)}")

    print("Generating embeddings — this may take a few minutes...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(all_chunks, embeddings)

    print("Saving vector store...")
    vector_store.save_local(str(VECTOR_STORE_DIR))
    print(f"Done. Index saved to {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    build_index()