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
COLLECTION_CUSTOMER = "customer_documents"
COLLECTION_EMPLOYEE = "employee_documents"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "openai"
TOP_K = int(os.getenv("TOP_K", "5"))

CUSTOMER_SYSTEM_PROMPT = """You are a helpful customer assistant for AutoGroup Motors (branded as HR CARs), a car resale dealership located at Gedimino pr. 45, Vilnius, Lithuania.

Rules:
1. Answer based on the provided context. Use reasoning — if the context describes a service (e.g. delivery options, financing), you can confirm that the dealership offers it.
2. If the context genuinely contains no relevant information, say "I don't have enough information to answer that."
3. Always answer in the same language as the user's question. You MUST support Lithuanian (lietuvių kalba) — if the user writes in Lithuanian, respond fully in Lithuanian.
4. Be concise and helpful. Provide specific details (prices, timelines, contacts) when available in the context.
5. When the user refers to previous messages in the conversation, use the conversation history to understand what they mean.
6. You are speaking to a customer. Do NOT reveal any internal employee information such as salaries, benefits, or internal policies."""

EMPLOYEE_SYSTEM_PROMPT = """You are an internal assistant for AutoGroup Motors (branded as HR CARs) employees, located at Gedimino pr. 45, Vilnius, Lithuania.

Rules:
1. Answer based on the provided context. Use reasoning — if the context describes a policy, benefit, or procedure, you can confirm it.
2. If the context genuinely contains no relevant information, say "I don't have enough information to answer that."
3. Always answer in the same language as the user's question. You MUST support Lithuanian (lietuvių kalba) — if the user writes in Lithuanian, respond fully in Lithuanian.
4. Be concise and helpful. Provide specific details (salary bands, benefit amounts, policy details) when available in the context.
5. When the user refers to previous messages in the conversation, use the conversation history to understand what they mean.
6. You are speaking to an employee. You have access to all company documents including confidential HR information."""


def get_retriever():
    """Initialize ChromaDB client, both collections, and embedding model."""
    if not os.path.exists(CHROMA_DB_DIR):
        print("Error: Vector store not found. Run 'python ingest.py' first.")
        sys.exit(1)

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collections = {}
    try:
        collections["customer"] = client.get_collection(name=COLLECTION_CUSTOMER)
        collections["employee"] = client.get_collection(name=COLLECTION_EMPLOYEE)
    except (ValueError, Exception):
        print("Error: Collections not found. Run 'python ingest.py' first.")
        sys.exit(1)

    model = SentenceTransformer(EMBEDDING_MODEL)
    return collections, model


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


def ask_ollama(messages: list[dict]) -> str:
    """Send messages to Ollama chat API and return the generated response."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    except requests.ConnectionError:
        return "Error: Cannot connect to Ollama. Make sure it's running (ollama serve)."
    except requests.Timeout:
        return "Error: Ollama request timed out. The model may be loading — try again."
    except requests.RequestException as e:
        return f"Error communicating with Ollama: {e}"


def ask_openai(messages: list[dict]) -> str:
    """Send messages to OpenAI and return the generated response."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error communicating with OpenAI: {e}"


def ask_llm(messages: list[dict]) -> str:
    """Route messages to the configured LLM provider."""
    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your-openai-api-key-here":
            return "Error: OPENAI_API_KEY is not set in .env file."
        return ask_openai(messages)
    return ask_ollama(messages)


def answer_question(
    question: str,
    collections: dict,
    model: SentenceTransformer,
    mode: str = "customer",
    history: list[dict] | None = None,
) -> dict:
    """Full RAG pipeline: retrieve context, build prompt, generate answer."""
    collection = collections.get(mode, collections["customer"])
    system_prompt = EMPLOYEE_SYSTEM_PROMPT if mode == "employee" else CUSTOMER_SYSTEM_PROMPT

    context, sources = retrieve_chunks(question, collection, model)

    # Build chat messages with system prompt, history, and current question
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history (last 3 exchanges)
    if history:
        for entry in history[-3:]:
            messages.append({"role": "user", "content": entry["question"]})
            messages.append({"role": "assistant", "content": entry["answer"]})

    # Current question with retrieved context
    user_msg = f"Context:\n{context}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_msg})

    answer = ask_llm(messages)
    return {"answer": answer, "sources": sources}


def main():
    print("Loading RAG chatbot...")
    collections, model = get_retriever()
    print(f"Ready! Using model '{OLLAMA_MODEL}' via Ollama.")

    mode = input("Select mode (customer/employee) [customer]: ").strip().lower()
    if mode not in ("customer", "employee"):
        mode = "customer"
    print(f"Mode: {mode}")
    print("Type your question and press Enter. Type 'exit' to quit.\n")

    history = []

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

        result = answer_question(question, collections, model, mode=mode, history=history)
        history.append({"question": question, "answer": result["answer"]})
        print(f"\nAssistant: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])}\n")


if __name__ == "__main__":
    main()
