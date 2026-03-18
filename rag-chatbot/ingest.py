"""
ingest.py — Document ingestion pipeline for the RAG chatbot.

Reads .txt and .md files from ./documents/, splits them into chunks,
generates embeddings with all-MiniLM-L6-v2, and stores everything in ChromaDB.
"""

import os
import sys
import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Load environment variables from .env
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# --- Configuration (reads from .env with sensible defaults) ---
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_CUSTOMER = "customer_documents"
COLLECTION_EMPLOYEE = "employee_documents"

# Documents that contain sensitive employee information — excluded from customer collection
EMPLOYEE_ONLY_FILES = {
    "Employee-Compensation-And-Pay-Structure.txt",
    "Employee-Health-And-Benefits-Package.txt",
}
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


def load_documents(documents_dir: str) -> list[dict]:
    """Read all .txt and .md files from the documents directory."""
    documents = []
    if not os.path.exists(documents_dir):
        print(f"Error: Documents directory '{documents_dir}' does not exist.")
        sys.exit(1)

    for filename in sorted(os.listdir(documents_dir)):
        if filename.endswith((".txt", ".md")):
            filepath = os.path.join(documents_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            if content.strip():
                documents.append({"filename": filename, "content": content})

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, text in enumerate(splits):
            chunks.append({
                "text": text,
                "metadata": {
                    "source": doc["filename"],
                    "chunk_index": i,
                },
            })

    return chunks


def store_in_chromadb(chunks: list[dict], model: SentenceTransformer, collection_name: str) -> None:
    """Generate embeddings and store chunks in ChromaDB. Clears existing data first."""
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Idempotent: delete the collection if it exists, then recreate it
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # Collection doesn't exist yet
    collection = client.get_or_create_collection(name=collection_name)

    # Generate embeddings for all chunks at once
    texts = [chunk["text"] for chunk in chunks]
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # Prepare data for ChromaDB
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [chunk["metadata"] for chunk in chunks]

    # Add to collection in a single batch
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def main():
    # Step 1: Load all documents
    print(f"Reading documents from: {DOCUMENTS_DIR}")
    all_documents = load_documents(DOCUMENTS_DIR)
    if not all_documents:
        print("No .txt or .md files found in the documents folder.")
        print("Add some files and try again.")
        sys.exit(1)
    print(f"Found {len(all_documents)} document(s): {[d['filename'] for d in all_documents]}")

    # Step 2: Split into customer and employee document sets
    customer_docs = [d for d in all_documents if d["filename"] not in EMPLOYEE_ONLY_FILES]
    employee_docs = all_documents  # employees see everything

    print(f"\nCustomer documents ({len(customer_docs)}): {[d['filename'] for d in customer_docs]}")
    print(f"Employee documents ({len(employee_docs)}): {[d['filename'] for d in employee_docs]}")
    print(f"Employee-only files: {EMPLOYEE_ONLY_FILES}")

    # Step 3: Chunk both sets
    customer_chunks = chunk_documents(customer_docs)
    employee_chunks = chunk_documents(employee_docs)
    print(f"\nCustomer chunks: {len(customer_chunks)} (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"Employee chunks: {len(employee_chunks)}")

    # Step 4: Load the embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Step 5: Store both collections in ChromaDB
    print(f"\n--- Ingesting customer collection ---")
    store_in_chromadb(customer_chunks, model, COLLECTION_CUSTOMER)

    print(f"\n--- Ingesting employee collection ---")
    store_in_chromadb(employee_chunks, model, COLLECTION_EMPLOYEE)

    # Summary
    print("\n--- Ingestion Complete ---")
    print(f"Vector store location: {CHROMA_DB_DIR}")
    print(f"\nCustomer collection '{COLLECTION_CUSTOMER}': {len(customer_chunks)} chunks from {len(customer_docs)} docs")
    for doc in customer_docs:
        doc_chunks = [c for c in customer_chunks if c["metadata"]["source"] == doc["filename"]]
        print(f"  {doc['filename']}: {len(doc_chunks)} chunks")
    print(f"\nEmployee collection '{COLLECTION_EMPLOYEE}': {len(employee_chunks)} chunks from {len(employee_docs)} docs")
    for doc in employee_docs:
        doc_chunks = [c for c in employee_chunks if c["metadata"]["source"] == doc["filename"]]
        print(f"  {doc['filename']}: {len(doc_chunks)} chunks")


if __name__ == "__main__":
    main()
