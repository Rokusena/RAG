"""
query.py — Terminal-based RAG chatbot.

Embeds user questions, retrieves relevant chunks from ChromaDB,
and generates answers using Ollama (llama3.2).
"""

import os
import sys
import chromadb
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables from .env
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# --- Configuration (reads from .env with sensible defaults) ---
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "rag_documents"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "openai"
TOP_K = int(os.getenv("TOP_K", "5"))

SYSTEM_PROMPT = """You are a helpful assistant for HR CARs, a car resale dealership in Lithuania.
Answer the question based ONLY on the provided context. If the context doesn't
contain enough information, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}"""


def get_retriever():
    """Initialize ChromaDB client and embedding model."""
    if not os.path.exists(CHROMA_DB_DIR):
        print("Error: Vector store not found. Run 'python ingest.py' first.")
        sys.exit(1)

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except (ValueError, Exception):
        print("Error: Collection not found. Run 'python ingest.py' first.")
        sys.exit(1)

    model = SentenceTransformer(EMBEDDING_MODEL)
    return collection, model


def retrieve_chunks(question: str, collection, model: SentenceTransformer) -> tuple[str, list[str]]:
    """Embed the question and retrieve the top-K most similar chunks."""
    query_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K,
    )

    # Extract chunk texts and source filenames
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    sources = list({m["source"] for m in metadatas})

    context = "\n\n".join(documents)
    return context, sources


def ask_ollama(prompt: str) -> str:
    """Send the prompt to Ollama and return the generated response."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure it's running (ollama serve)."
    except requests.Timeout:
        return "Error: Ollama request timed out. The model may be loading — try again."
    except requests.RequestException as e:
        return f"Error communicating with Ollama: {e}"


def ask_openai(prompt: str) -> str:
    """Send the prompt to OpenAI and return the generated response."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error communicating with OpenAI: {e}"


def ask_llm(prompt: str) -> str:
    """Route the prompt to the configured LLM provider."""
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your-openai-api-key-here":
            return "Error: OPENAI_API_KEY is not set in .env file."
        return ask_openai(prompt)
    return ask_ollama(prompt)


def answer_question(question: str, collection, model: SentenceTransformer) -> dict:
    """Full RAG pipeline: retrieve context, build prompt, generate answer."""
    context, sources = retrieve_chunks(question, collection, model)
    prompt = SYSTEM_PROMPT.format(context=context, question=question)
    answer = ask_llm(prompt)
    return {"answer": answer, "sources": sources}


def main():
    print("Loading RAG chatbot...")
    collection, model = get_retriever()
    print(f"Ready! Using model '{OLLAMA_MODEL}' via Ollama.")
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() == "exit":
            print("Goodbye!")
            break

        result = answer_question(question, collection, model)
        print(f"\nAssistant: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])}\n")


if __name__ == "__main__":
    main()
