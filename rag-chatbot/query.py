"""
query.py — Terminal-based RAG chatbot.

Embeds user questions, retrieves relevant chunks from ChromaDB,
and generates answers using Ollama (qwen3.5:9B with /no_think mode).
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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:9B")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "openai"
TOP_K = int(os.getenv("TOP_K", "5"))

CUSTOMER_SYSTEM_PROMPT = """You are the customer-facing assistant for AutoGroup Motors (brand name: HR CARs), Gedimino pr. 45, Vilnius, Lithuania.

Rules:
1. Answer ONLY using the provided context. Do NOT add information from outside the context.
2. Copy exact numbers, prices, dates, percentages, and names from the context — never round, approximate, or paraphrase numerical data.
3. If the context has no relevant information, reply exactly: "I don't have enough information to answer that."
4. Reply in the same language the user writes in.
5. Answer concisely but always include every specific number, price, percentage, and condition found in the context. Do not omit details to save space.
6. Use conversation history to resolve references like "that car" or "the previous one".
7. NEVER reveal employee-only information (salaries, benefits, internal policies)."""

EMPLOYEE_SYSTEM_PROMPT = """You are the internal assistant for AutoGroup Motors (brand name: HR CARs) employees, Gedimino pr. 45, Vilnius, Lithuania.

Rules:
1. Answer ONLY using the provided context. Do NOT add information from outside the context.
2. Copy exact numbers, prices, dates, percentages, salary bands, and benefit amounts from the context — never round, approximate, or paraphrase numerical data.
3. If the context has no relevant information, reply exactly: "I don't have enough information to answer that."
4. Reply in the same language the user writes in.
5. Answer concisely but always include every specific number, price, percentage, and condition found in the context. Do not omit details to save space.
6. Use conversation history to resolve references like "that policy" or "the previous question".
7. You have full access to all company documents including confidential HR information."""


# --- FAQ: instant answers for common questions ---
CUSTOMER_FAQ = [
    {
        "patterns": ["return policy", "return a car", "can i return", "what is the return policy", "returning a vehicle"],
        "answer": "For in-person purchases, we offer a voluntary 3-day satisfaction guarantee. For distance/online purchases, you have a 14-day cooling-off period. The vehicle must have less than 300 km added, be in original condition, and all documentation/keys returned. Refunds are processed within 14 business days. Contact: complaints@autogroupmotors.lt or +370 5 123 4573.",
        "sources": ["Returns-Refunds-And-Dispute-Resolution.txt"],
    },
    {
        "patterns": ["warranty", "what warranty", "warranty coverage", "is there a warranty"],
        "answer": "Every pre-owned vehicle comes with a minimum 3-month / 5,000 km powertrain warranty. We also offer extended plans: Bronze (12 months, EUR 299), Silver (18 months, EUR 549), and Gold (24 months, EUR 899). New vehicles carry the full manufacturer warranty. Contact: service@autogroupmotors.lt or +370 5 123 4567.",
        "sources": ["Warranty-And-After-Sales-Policy.txt"],
    },
    {
        "patterns": ["test drive", "can i test drive", "how to book a test drive", "schedule a test drive"],
        "answer": "You can book a test drive online at www.autogroupmotors.lt/test-drive, by calling +370 5 123 4567, or walk in. You need a valid driving license and ID. Each slot is 30 minutes (15-20 min driving). Available Mon-Fri 9:00-17:00 and Sat 9:00-14:00. Insurance excess is EUR 500 in case of negligence.",
        "sources": ["Test-Drive-Policy-And-Procedure.txt"],
    },
    {
        "patterns": ["financing", "loan", "auto loan", "financing options", "can i finance", "leasing"],
        "answer": "We partner with SEB, Swedbank, Luminor, and Šiaulių Bankas. Auto loan rates start from 4.9% APR, terms 12-84 months, down payment as low as 10%. Pre-approval within 2 business hours. We also offer business leasing (24-60 months) and a First-Time Buyer Program (6.5% APR, ages 18-25). Contact: finance@autogroupmotors.lt or +370 5 123 4568.",
        "sources": ["Financing-And-Leasing-Options.txt"],
    },
    {
        "patterns": ["oil change", "oil change price", "how much is an oil change"],
        "answer": "Oil change prices: Conventional oil EUR 49, Synthetic oil EUR 79, Diesel vehicles EUR 89. All include up to 5-6L of oil and oil filter. Prices include 21% VAT. Book at service@autogroupmotors.lt or +370 5 123 4567.",
        "sources": ["Service-And-Maintenance-Price-List.txt"],
    },
    {
        "patterns": ["referral", "referral program", "loyalty program", "loyalty", "refer a friend"],
        "answer": "Our loyalty program has 3 tiers: Bronze (10% labor discount), Silver (15% + free Small Service/year), Gold (20% labor + 10% parts + free Full Service/year). Referral rewards: you get EUR 200 service credit or EUR 150 cash, the referred customer gets EUR 100 service credit. No limit on referrals. Contact: loyalty@autogroupmotors.lt or +370 5 123 4572.",
        "sources": ["Customer-Loyalty-And-Referral-Program.txt"],
    },
    {
        "patterns": ["where are you located", "address", "location", "where is the dealership"],
        "answer": "AutoGroup Motors (HR CARs) is located at Gedimino pr. 45, Vilnius, Lithuania. Open Mon-Fri 9:00-18:00, Sat 9:00-15:00. Contact: +370 5 123 4567 or sales@autogroupmotors.lt.",
        "sources": ["Vehicle-Stock-And-Inventory.txt"],
    },
    {
        "patterns": ["service price", "maintenance cost", "how much does service cost", "service packages"],
        "answer": "Service packages: Small Service EUR 99 (petrol) / EUR 119 (diesel) every 15,000 km. Full Service EUR 199 / EUR 239 every 30,000 km. Major Service EUR 349-599 every 60,000 km. OBD diagnostic EUR 29 (waived with repair). All prices include 21% VAT. Book at service@autogroupmotors.lt.",
        "sources": ["Service-And-Maintenance-Price-List.txt"],
    },
]

EMPLOYEE_FAQ = [
    {
        "patterns": ["salary", "pay", "how much do i earn", "compensation", "salary bands"],
        "answer": "Salary bands vary by role. Examples: General Manager EUR 4,500-6,000/mo, Sales Consultant EUR 1,600-2,200/mo + commission, Senior Mechanic EUR 1,800-2,400/mo, Accountant EUR 1,800-2,400/mo. Sales commission: 1.5% of vehicle gross profit at target margin. Annual bonus: up to 1.5 months' salary. Salary reviews happen in March. Contact: hr@autogroupmotors.lt.",
        "sources": ["Employee-Compensation-And-Pay-Structure.txt"],
    },
    {
        "patterns": ["overtime", "overtime pay", "overtime policy", "overtime rate", "overtime hours"],
        "answer": "Overtime must be pre-approved by your manager. Rates: weekday overtime 1.5x hourly rate, Saturday 1.5x, Sunday and public holidays 2.0x. Maximum 8 hours overtime per week and 180 hours per calendar year per Lithuanian labour law.",
        "sources": ["Employee-Compensation-And-Pay-Structure.txt"],
    },
]


def _match_faq(question: str, mode: str) -> dict | None:
    """Check if the question matches any FAQ pattern. Returns the FAQ entry or None."""
    q = question.lower().strip()
    faq_list = EMPLOYEE_FAQ + CUSTOMER_FAQ if mode == "employee" else CUSTOMER_FAQ
    for entry in faq_list:
        for pattern in entry["patterns"]:
            if pattern in q or q in pattern:
                return entry
    return None


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
                "options": {"num_predict": 512},
            },
            timeout=180,
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
    # Check FAQ first for instant answers
    faq_match = _match_faq(question, mode)
    if faq_match:
        return {"answer": faq_match["answer"], "sources": faq_match["sources"]}

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
    # /no_think disables qwen3's chain-of-thought — faster for RAG where the answer is in the chunks
    user_msg = f"/no_think\n\nContext:\n{context}\n\nQuestion: {question}"
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
