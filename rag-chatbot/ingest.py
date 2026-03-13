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
COLLECTION_NAME = "rag_documents"
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


def store_in_chromadb(chunks: list[dict], model: SentenceTransformer) -> None:
    """Generate embeddings and store chunks in ChromaDB. Clears existing data first."""
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Idempotent: delete the collection if it exists, then recreate it
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # Collection doesn't exist yet
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

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
    # Step 1: Load documents
    print(f"Reading documents from: {DOCUMENTS_DIR}")
    documents = load_documents(DOCUMENTS_DIR)
    if not documents:
        print("No .txt or .md files found in the documents folder.")
        print("Add some files and try again.")
        sys.exit(1)
    print(f"Found {len(documents)} document(s): {[d['filename'] for d in documents]}")

    # Step 2: Chunk documents
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    # Step 3: Load the embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Step 4: Store in ChromaDB
    store_in_chromadb(chunks, model)

    # Summary
    print("\n--- Ingestion Complete ---")
    print(f"Documents processed: {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")
    print(f"Vector store location: {CHROMA_DB_DIR}")
    for doc in documents:
        doc_chunks = [c for c in chunks if c["metadata"]["source"] == doc["filename"]]
        print(f"  {doc['filename']}: {len(doc_chunks)} chunks")


if __name__ == "__main__":
    main()
