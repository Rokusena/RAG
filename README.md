# Document Q&A Chatbot

A RAG-powered chatbot that answers questions from your own documents using vector search and LLM generation. Supports OpenAI or fully local inference via Ollama.

## Features

- **Multi-format ingestion** — `.txt`, `.md`, and `.pdf` documents
- **Dual LLM support** — OpenAI API (`gpt-4o-mini`) or local Ollama (`qwen3.5:9B`)
- **Web UI + REST API** — FastAPI server with chat interface at `localhost:8000`
- **Access control** — Customer mode (public docs only) and Employee mode (includes confidential data)
- **FAQ layer** — Common questions answered instantly without LLM latency
- **Evaluation framework** — 20-question test suite with expected vs actual answer comparison
- **Configurable** — chunk size, overlap, top-K, model, and provider all via `.env`

## Quick Start

**1. Install dependencies**
```bash
cd rag-chatbot
pip install -r requirements.txt
```

**2. Configure `.env`**
```bash
# For OpenAI:
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# For local Ollama:
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3.5:9B
```
If using Ollama, install it from [ollama.com](https://ollama.com) and run `ollama pull qwen3.5:9B`.

**3. Add documents**

Drop `.txt`, `.md`, or `.pdf` files into `rag-chatbot/documents/`.

**4. Ingest**
```bash
python ingest.py
```

**5. Run**
```bash
python api.py          # Web UI at http://localhost:8000
python query.py        # Terminal chat
```

## Architecture

```
documents/  ──→  ingest.py  ──→  ChromaDB (vector store)
(.txt .md .pdf)    │                     │
                   │ chunk + embed       │ similarity search
                   ▼                     ▼
             Sentence Transformer    Top-K chunks
             (all-MiniLM-L6-v2)         │
                                        ▼
User question ──→ FAQ check ──→ LLM (Ollama / OpenAI) ──→ Answer + Sources
                  (instant)        │
                                   └─ system prompt + context + history
```

**Ingestion:** Documents are split into overlapping chunks, embedded with `all-MiniLM-L6-v2`, and stored in ChromaDB. Two collections are maintained — customer (public) and employee (all docs).

**Query:** Questions are checked against FAQ patterns first. On miss, the question is embedded and matched against stored chunks via cosine similarity. The top-K chunks are passed as context to the LLM, which generates a grounded answer.

## Configuration

All settings live in `rag-chatbot/.env` and are loaded by `config.py`.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` or `openai` |
| `OLLAMA_MODEL` | `qwen3.5:9B` | Ollama model name |
| `OPENAI_API_KEY` | — | Required when `LLM_PROVIDER=openai` |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer for embeddings |
| `TOP_K` | `5` | Chunks retrieved per query |
| `CHUNK_SIZE` | `400` | Characters per chunk |
| `CHUNK_OVERLAP` | `60` | Overlap between chunks |

## Evaluation

```bash
python eval.py    # generates evals/eval_<nr>.txt
```

Runs 20 questions (15 customer, 5 employee) through the full pipeline and saves a report comparing expected vs actual answers, retrieval precision, and FAQ coverage. Reports auto-increment for tracking changes across iterations.

See [EVAL.md](EVAL.md) for methodology, model comparison results, and known issues.

**Headline result:** Retrieval precision is 97.5% across 20 questions. Cosine similarity was dropped as an eval metric after finding that smarter models score lower by paraphrasing — despite giving better real-world answers.

## Tech Stack

ChromaDB · Sentence Transformers · LangChain text splitters · FastAPI · Ollama / OpenAI · pypdf
