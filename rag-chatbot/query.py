"""
query.py — Terminal-based RAG chatbot.

Embeds user questions, retrieves relevant chunks from ChromaDB,
and generates answers using the configured LLM provider (Ollama or OpenAI).
"""

import os
import sys
import chromadb
import requests
from openai import OpenAI

from config import (
    CHROMA_DB_DIR,
    COLLECTION_CUSTOMER,
    COLLECTION_EMPLOYEE,
    LLM_PROVIDER,
    OLLAMA_MODEL,
    OLLAMA_URL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_OLLAMA_CUSTOMER,
    SYSTEM_PROMPT_OLLAMA_EMPLOYEE,
    TOP_K,
    active_model_name,
)
from embeddings import embed_query


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
        "patterns": ["overtime", "overtime pay", "overtime policy", "overtime rate", "overtime hours"],
        "answer": "Overtime must be pre-approved by your manager. Rates: weekday overtime 1.5x hourly rate, Saturday 1.5x, Sunday and public holidays 2.0x. Maximum 8 hours overtime per week and 180 hours per calendar year per Lithuanian labour law.",
        "sources": ["Employee-Compensation-And-Pay-Structure.txt"],
    },
    {
        "patterns": ["salary", "how much do i earn", "compensation", "salary bands", "pay structure", "pay scale"],
        "answer": "Salary bands vary by role. Examples: General Manager EUR 4,500-6,000/mo, Sales Consultant EUR 1,600-2,200/mo + commission, Senior Mechanic EUR 1,800-2,400/mo, Accountant EUR 1,800-2,400/mo. Sales commission: 1.5% of vehicle gross profit at target margin. Annual bonus: up to 1.5 months' salary. Salary reviews happen in March. Contact: hr@autogroupmotors.lt.",
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


def get_retriever() -> dict:
    """Initialize ChromaDB client and return both collections."""
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

    return collections


def retrieve_chunks(question: str, collection) -> tuple[str, list[str], list[dict]]:
    """Embed the question and retrieve the top-K most similar chunks.

    Returns (joined_context, unique_sources, ranked_chunks). Each ranked chunk is
    {"source", "text", "distance"} in rank order — useful for eval diagnostics.
    """
    query_embedding = [embed_query(question)]

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results.get("distances", [[None] * len(documents)])[0]

    ranked_chunks = [
        {"source": m["source"], "text": doc, "distance": dist}
        for doc, m, dist in zip(documents, metadatas, distances)
    ]
    sources = list({m["source"] for m in metadatas})
    context = "\n\n".join(documents)
    return context, sources, ranked_chunks


def ask_ollama(messages: list[dict]) -> str:
    """Send messages to Ollama chat API and return the generated response."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "keep_alive": "30m",
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
        if not OPENAI_API_KEY:
            return "Error: OPENAI_API_KEY is not set in .env file."
        return ask_openai(messages)
    return ask_ollama(messages)


def answer_question(
    question: str,
    collections: dict,
    mode: str = "customer",
    history: list[dict] | None = None,
) -> dict:
    """Full RAG pipeline: retrieve context, build prompt, generate answer."""
    # Check FAQ first for instant answers
    faq_match = _match_faq(question, mode)
    if faq_match:
        return {"answer": faq_match["answer"], "sources": faq_match["sources"], "chunks": []}

    collection = collections.get(mode, collections["customer"])
    context, sources, chunks = retrieve_chunks(question, collection)

    # Select system prompt based on provider
    if LLM_PROVIDER == "openai":
        system_prompt = SYSTEM_PROMPT.replace("{mode}", mode)
    else:
        system_prompt = SYSTEM_PROMPT_OLLAMA_EMPLOYEE if mode == "employee" else SYSTEM_PROMPT_OLLAMA_CUSTOMER

    # Build chat messages — context goes in user message, not system message
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history (last 3 exchanges)
    if history:
        for entry in history[-3:]:
            messages.append({"role": "user", "content": entry["question"]})
            messages.append({"role": "assistant", "content": entry["answer"]})

    # /no_think disables qwen3's chain-of-thought — only used with Ollama
    if LLM_PROVIDER == "ollama":
        user_msg = f"/no_think\n\nContext:\n{context}\n\nQuestion: {question}"
    else:
        user_msg = f"Context:\n{context}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_msg})

    answer = ask_llm(messages)
    return {"answer": answer, "sources": sources, "chunks": chunks}


def main():
    print("Loading RAG chatbot...")
    collections = get_retriever()
    print(f"Ready! Using model '{active_model_name()}' via {LLM_PROVIDER}.")

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

        result = answer_question(question, collections, mode=mode, history=history)
        history.append({"question": question, "answer": result["answer"]})
        print(f"\nAssistant: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])}\n")


if __name__ == "__main__":
    main()
