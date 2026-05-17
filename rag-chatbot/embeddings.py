"""
embeddings.py — Embedding helper backed by OpenAI's embedding API.

The model is configured via EMBEDDING_MODEL in .env (default: text-embedding-3-small).
"""

from openai import OpenAI

from config import EMBEDDING_MODEL, OPENAI_API_KEY

_BATCH_SIZE = 100
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set in .env file.")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts. Batched to stay well under the OpenAI request limit."""
    if not texts:
        return []
    client = _get_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start:start + _BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
