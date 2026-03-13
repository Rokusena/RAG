"""
api.py — FastAPI server for the RAG chatbot.

Provides a POST /ask endpoint that accepts a question and returns
an answer with source references.
"""

import os
from contextlib import asynccontextmanager

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from query import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    SYSTEM_PROMPT,
    TOP_K,
    ask_llm,
    retrieve_chunks,
)

# --- Request / Response models ---

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: list[str]


# --- Application lifespan: load models once at startup ---

# Shared state loaded during startup
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding model and ChromaDB collection once at startup."""
    if not os.path.exists(CHROMA_DB_DIR):
        raise RuntimeError(
            "Vector store not found. Run 'python ingest.py' first."
        )

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        raise RuntimeError(
            "Collection not found. Run 'python ingest.py' first."
        )

    model = SentenceTransformer(EMBEDDING_MODEL)

    _state["collection"] = collection
    _state["model"] = model
    print(f"RAG API ready — collection has {collection.count()} chunks")
    yield
    _state.clear()


# --- FastAPI app ---

app = FastAPI(
    title="HR CARs RAG Chatbot API",
    description="Ask questions about HR CARs dealership documents.",
    lifespan=lifespan,
)

# CORS middleware — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Accept a question, retrieve relevant context, and return an LLM answer."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    collection = _state["collection"]
    model = _state["model"]

    # Retrieve relevant chunks
    context, sources = retrieve_chunks(question, collection, model)

    # Build prompt and call Ollama
    prompt = SYSTEM_PROMPT.format(context=context, question=question)
    answer = ask_llm(prompt)

    # Surface Ollama errors as 503
    if answer.startswith("Error:"):
        raise HTTPException(status_code=503, detail=answer)

    return AskResponse(answer=answer, sources=sources)


@app.get("/")
async def serve_ui():
    """Serve the chat UI."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(html_path, media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
