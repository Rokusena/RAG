# HR CARs RAG Chatbot

A local Retrieval-Augmented Generation chatbot for HR CARs, a car resale dealership in Lithuania. Everything runs locally — no external API keys needed.

## Tech Stack

- **Embeddings**: sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector Store**: ChromaDB (file-based, persisted in `./chroma_db/`)
- **Text Splitting**: LangChain `RecursiveCharacterTextSplitter`
- **LLM**: Ollama with `llama3.2`
- **API**: FastAPI

## Setup

### 1. Install Python dependencies

```bash
cd rag-chatbot
pip install -r requirements.txt
```

### 2. Install Ollama and pull the model

Download Ollama from [ollama.com](https://ollama.com) and install it.

```bash
# Pull the llama3.2 model (~2GB)
ollama pull llama3.2
```s

Make sure Ollama is running (it starts automatically on install, or run `ollama serve`).

### 3. Add your documents

Place `.txt` and/or `.md` files in the `documents/` folder. These are the source files the chatbot will use to answer questions.

### 4. Run ingestion

```bash
python ingest.py
```

This reads all documents, splits them into chunks, generates embeddings, and stores everything in ChromaDB. Re-running this command will clear and rebuild the entire vector store.

### 5. Chat from the terminal

```bash
python query.py
```

Type questions and get answers. Type `exit` to quit.

### 6. Run the API server

```bash
python api.py
```

The server starts at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

**Endpoint:**

```
POST /ask
Content-Type: application/json

{"question": "What cars do you have in stock?"}
```

**Response:**

```json
{
  "answer": "Based on the documents...",
  "sources": ["Vehicle-Inventory.txt", "FAQ.txt"]
}
```
