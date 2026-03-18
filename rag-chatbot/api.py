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
    COLLECTION_CUSTOMER,
    COLLECTION_EMPLOYEE,
    EMBEDDING_MODEL,
    answer_question,
)

# --- Request / Response models ---

class HistoryEntry(BaseModel):
    question: str
    answer: str

class AskRequest(BaseModel):
    question: str
    mode: str = "customer"
    history: list[HistoryEntry] = []

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
        customer_col = client.get_collection(name=COLLECTION_CUSTOMER)
        employee_col = client.get_collection(name=COLLECTION_EMPLOYEE)
    except Exception:
        raise RuntimeError(
            "Collections not found. Run 'python ingest.py' first."
        )

    model = SentenceTransformer(EMBEDDING_MODEL)

    _state["collections"] = {"customer": customer_col, "employee": employee_col}
    _state["model"] = model
    print(f"RAG API ready — customer: {customer_col.count()} chunks, employee: {employee_col.count()} chunks")
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

    collections = _state["collections"]
    model = _state["model"]
    mode = request.mode if request.mode in ("customer", "employee") else "customer"

    # Convert history to dicts for query module
    history = [entry.model_dump() for entry in request.history]

    result = answer_question(question, collections, model, mode=mode, history=history)

    # Surface LLM errors as 503
    if result["answer"].startswith("Error:"):
        raise HTTPException(status_code=503, detail=result["answer"])

    return AskResponse(answer=result["answer"], sources=result["sources"])


@app.get("/")
async def serve_ui():
    """Serve the chat UI."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(html_path, media_type="text/html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
