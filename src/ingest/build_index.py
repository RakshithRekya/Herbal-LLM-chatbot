from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from deep_translator import GoogleTranslator
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.config import RAW_PDFS_DIR, VECTOR_STORE_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from src.ingest.translator import is_greek


BOILERPLATE_PHRASES = [
    "αποξηραμένα βότανα μπορούν να αποθηκευτούν",
    "ξήρανση των βοτάνων σε πολύ υψηλή θερμοκρασία",
    "γεύση προέρχεται από έλαια στα κυτταρικά τοιχώματα",
    "Consultation with a specialist is considered necessary",
    "information provided has been drawn from academic books",
]


def is_boilerplate(text):
    return any(phrase.lower() in text.lower() for phrase in BOILERPLATE_PHRASES)


def translate_chunk(text):
    try:
        if len(text) <= 4500:
            return GoogleTranslator(source='el', target='en').translate(text)
        parts = []
        words = text.split()
        current = ""
        for word in words:
            if len(current) + len(word) + 1 < 4500:
                current += " " + word
            else:
                parts.append(GoogleTranslator(source='el', target='en').translate(current.strip()))
                current = word
        if current:
            parts.append(GoogleTranslator(source='el', target='en').translate(current.strip()))
        return " ".join(parts)
    except Exception as e:
        print(f"  Translation error: {e}")
        return text  # fallback to original if translation fails


def load_documents():
    documents = []

    pdf_loader = PyPDFDirectoryLoader(str(RAW_PDFS_DIR), recursive=True)
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


def build_index(test_mode=False):
    print("Loading documents...")
    documents = load_documents()

    if not documents:
        print("No documents found.")
        return

    # In test mode only process first 15 documents
    if test_mode:
        documents = documents[:15]
        print(f"TEST MODE: Processing first 15 documents only.")

    print(f"Total: {len(documents)} documents loaded.")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # Remove boilerplate
    chunks = [c for c in chunks if not is_boilerplate(c.page_content)]
    print(f"After removing boilerplate: {len(chunks)} chunks remaining.")

    # Translate Greek chunks to English
    print("Translating chunks to English...")
    translated_chunks = []
    for i, chunk in enumerate(chunks):
        if is_greek(chunk.page_content):
            translated = translate_chunk(chunk.page_content)
            translated_chunks.append(Document(
                page_content=translated,
                metadata={**chunk.metadata, "language": "en", "translated": True}
            ))
            print(f"  [{i+1}/{len(chunks)}] Translated.")
        else:
            # Already English, keep as is
            translated_chunks.append(Document(
                page_content=chunk.page_content,
                metadata={**chunk.metadata, "language": "en"}
            ))

    print(f"Total chunks in index: {len(translated_chunks)}")

    print("Generating embeddings...")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(translated_chunks, embeddings)

    print("Saving vector store...")
    vector_store.save_local(str(VECTOR_STORE_DIR))
    print(f"Done. Index saved to {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run in test mode with 15 documents")
    args = parser.parse_args()
    build_index(test_mode=args.test)