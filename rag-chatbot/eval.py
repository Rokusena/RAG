"""
eval.py — Evaluation script for the RAG chatbot.

Measures:
1. Retrieval precision: did the right source documents get pulled?
2. Answer correctness: cosine similarity between expected and actual answers.
3. FAQ hit rate: did FAQ-eligible questions get instant answers?

Usage:
    python eval.py
"""

import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer

# Add parent directory context
sys.path.insert(0, os.path.dirname(__file__))

from query import get_retriever, answer_question, _match_faq

# --- Evaluation Dataset ---
# Each entry: question, expected answer (key facts), expected source files, mode

EVAL_DATASET = [
    # Customer questions
    {
        "question": "What is the return policy for vehicles?",
        "expected_answer": "In-person purchases have a 3-day satisfaction guarantee. Distance/online purchases have a 14-day cooling-off period. Vehicle must not be driven more than 300 km beyond delivery odometer, must be in original condition with all documentation returned.",
        "expected_sources": ["Returns-Refunds-And-Dispute-Resolution.txt"],
        "mode": "customer",
    },
    {
        "question": "What warranty do pre-owned vehicles come with?",
        "expected_answer": "Every pre-owned vehicle comes with a minimum 3-month or 5,000-kilometer powertrain warranty covering engine, transmission, differential, drive axles, and transfer case.",
        "expected_sources": ["Warranty-And-After-Sales-Policy.txt"],
        "mode": "customer",
    },
    {
        "question": "What are the extended warranty options and prices?",
        "expected_answer": "Bronze Plan: powertrain, 12 months or 20,000 km, EUR 299. Silver Plan: powertrain plus electrical, AC, steering, 18 months or 30,000 km, EUR 549. Gold Plan: bumper-to-bumper, 24 months or 40,000 km, EUR 899. All include 24/7 roadside assistance.",
        "expected_sources": ["Warranty-And-After-Sales-Policy.txt"],
        "mode": "customer",
    },
    {
        "question": "How can I schedule a test drive?",
        "expected_answer": "Test drives can be booked online at www.autogroupmotors.lt/test-drive, by phone at +370 5 123 4567, or walk-in. Available Mon-Fri 9:00-17:00 and Sat 9:00-14:00. Each slot is 30 minutes. You need a valid driving license and ID.",
        "expected_sources": ["Test-Drive-Policy-And-Procedure.txt"],
        "mode": "customer",
    },
    {
        "question": "What financing options are available?",
        "expected_answer": "Auto loans through SEB, Swedbank, Luminor, and Šiaulių Bankas starting from 4.9% APR. Loan terms 12-84 months, down payment as low as 10%. Business leasing available 24-60 months. First-Time Buyer Program at 6.5% APR for ages 18-25.",
        "expected_sources": ["Financing-And-Leasing-Options.txt"],
        "mode": "customer",
    },
    {
        "question": "How much does an oil change cost?",
        "expected_answer": "Conventional oil EUR 49, synthetic oil EUR 79, diesel vehicles EUR 89. Includes up to 5-6 liters of oil and oil filter. Prices include 21% VAT.",
        "expected_sources": ["Service-And-Maintenance-Price-List.txt"],
        "mode": "customer",
    },
    {
        "question": "What SUVs do you have in stock?",
        "expected_answer": "Current SUV stock includes Volvo XC60 EUR 42,500, BMW X3 EUR 39,900, Mercedes GLC EUR 37,800, Toyota RAV4 EUR 34,200, Hyundai Tucson EUR 33,900, VW Tiguan EUR 27,500, Mazda CX-5 EUR 28,700, Land Rover Discovery Sport EUR 26,900.",
        "expected_sources": ["Vehicle-Stock-And-Inventory.txt"],
        "mode": "customer",
    },
    {
        "question": "What is the referral program?",
        "expected_answer": "When you refer a new customer who completes a purchase, you receive EUR 200 service credit or EUR 150 cash. The referred customer gets EUR 100 service credit. No limit on referrals.",
        "expected_sources": ["Customer-Loyalty-And-Referral-Program.txt"],
        "mode": "customer",
    },
    {
        "question": "What electric vehicles do you sell?",
        "expected_answer": "Tesla Model 3 Long Range EUR 39,900, VW ID.4 EUR 34,500, BMW iX1 EUR 44,200, Volvo XC40 Recharge EUR 37,800, Mercedes EQA 250 EUR 33,500.",
        "expected_sources": ["Vehicle-Stock-And-Inventory.txt"],
        "mode": "customer",
    },
    {
        "question": "How does the loyalty program work?",
        "expected_answer": "Three tiers: Bronze (1st purchase, 10% labor discount), Silver (2nd purchase, 15% labor discount, free Small Service/year), Gold (3rd purchase, 20% labor + 10% parts discount, free Full Service/year, dedicated account manager).",
        "expected_sources": ["Customer-Loyalty-And-Referral-Program.txt"],
        "mode": "customer",
    },
    {
        "question": "How much does a full service cost?",
        "expected_answer": "Full Service costs EUR 199 for petrol and EUR 239 for diesel vehicles, recommended every 30,000 km or 24 months. Includes oil and filter change, air and cabin filter replacement, spark plug inspection, coolant check, transmission fluid check, drive belt inspection, and diagnostic scan.",
        "expected_sources": ["Service-And-Maintenance-Price-List.txt"],
        "mode": "customer",
    },
    {
        "question": "Can I reserve a vehicle?",
        "expected_answer": "You can reserve any vehicle for up to 72 hours with a refundable deposit of EUR 500. Reservations can be made online, by phone, or in person. Deposit refunded in full if purchase is not completed.",
        "expected_sources": ["Vehicle-Stock-And-Inventory.txt"],
        "mode": "customer",
    },
    {
        "question": "What is covered under defective vehicle claims?",
        "expected_answer": "If a defect is discovered within 6 months that existed at time of sale, the burden of proof lies with AutoGroup Motors. Options include free repair, replacement vehicle, partial refund, or full refund. Inspection scheduled within 5 business days.",
        "expected_sources": ["Returns-Refunds-And-Dispute-Resolution.txt"],
        "mode": "customer",
    },
    {
        "question": "Do you offer home test drives?",
        "expected_answer": "Yes, remote test drives are available within a 30 km radius of the dealership for a logistics fee of EUR 25. The fee is waived if you proceed with a purchase. Available by appointment only.",
        "expected_sources": ["Test-Drive-Policy-And-Procedure.txt"],
        "mode": "customer",
    },
    {
        "question": "Where is the dealership located?",
        "expected_answer": "AutoGroup Motors is located at Gedimino pr. 45, Vilnius, Lithuania. Open Monday to Friday 9:00-18:00 and Saturday 9:00-15:00.",
        "expected_sources": ["Vehicle-Stock-And-Inventory.txt"],
        "mode": "customer",
    },
    # Employee questions
    {
        "question": "What is the salary range for a sales consultant?",
        "expected_answer": "Sales Consultant base salary is EUR 1,600-2,200 gross per month plus commission. Commission is 1.5% of vehicle gross profit at target margin and 0.75% below target. Monthly volume bonus of EUR 200 for every 5 vehicles sold beyond 8.",
        "expected_sources": ["Employee-Compensation-And-Pay-Structure.txt"],
        "mode": "employee",
    },
    {
        "question": "What is the overtime policy?",
        "expected_answer": "Overtime must be pre-approved. Weekday overtime is 1.5x hourly rate, Saturday 1.5x, Sunday and public holidays 2.0x. Maximum 8 hours per week and 180 hours per calendar year.",
        "expected_sources": ["Employee-Compensation-And-Pay-Structure.txt"],
        "mode": "employee",
    },
    {
        "question": "When is salary paid each month?",
        "expected_answer": "Salary is paid on the 10th of the following month by bank transfer. Commission and bonus payments are also on the 10th of the month following calculation.",
        "expected_sources": ["Employee-Compensation-And-Pay-Structure.txt"],
        "mode": "employee",
    },
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(dot / norm) if norm > 0 else 0.0


def evaluate_retrieval_precision(actual_sources: list[str], expected_sources: list[str]) -> float:
    """What fraction of expected sources appear in retrieved sources."""
    if not expected_sources:
        return 1.0
    hits = sum(1 for s in expected_sources if s in actual_sources)
    return hits / len(expected_sources)


def main():
    print("=" * 60)
    print("RAG Chatbot Evaluation")
    print("=" * 60)

    print("\nLoading models and collections...")
    collections, model = get_retriever()

    total = len(EVAL_DATASET)
    retrieval_scores = []
    similarity_scores = []
    faq_hits = 0
    faq_eligible = 0

    print(f"Running {total} evaluation queries...\n")

    for i, entry in enumerate(EVAL_DATASET, 1):
        question = entry["question"]
        expected = entry["expected_answer"]
        expected_src = entry["expected_sources"]
        mode = entry["mode"]

        # Check if FAQ would handle this
        faq = _match_faq(question, mode)
        is_faq = faq is not None
        if is_faq:
            faq_eligible += 1
            faq_hits += 1

        # Get actual answer
        result = answer_question(question, collections, model, mode=mode)
        actual_answer = result["answer"]
        actual_sources = result["sources"]

        # Retrieval precision
        precision = evaluate_retrieval_precision(actual_sources, expected_src)
        retrieval_scores.append(precision)

        # Answer similarity (cosine similarity of sentence embeddings)
        emb_expected = model.encode([expected])[0]
        emb_actual = model.encode([actual_answer])[0]
        similarity = cosine_similarity(emb_expected, emb_actual)
        similarity_scores.append(similarity)

        # Print per-question results
        status = "OK" if precision == 1.0 and similarity > 0.5 else "WARN"
        faq_tag = " [FAQ]" if is_faq else ""
        print(f"  [{i:2d}/{total}] {status}{faq_tag} | Retrieval: {precision:.0%} | Similarity: {similarity:.2f}")
        print(f"         Q: {question}")
        if status == "WARN":
            print(f"         Expected src: {expected_src}")
            print(f"         Actual src:   {actual_sources}")
        print()

    # Summary
    avg_retrieval = np.mean(retrieval_scores) if retrieval_scores else 0
    avg_similarity = np.mean(similarity_scores) if similarity_scores else 0

    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Total questions:        {total}")
    print(f"  Avg retrieval precision: {avg_retrieval:.1%}")
    print(f"  Avg answer similarity:   {avg_similarity:.2f}")
    print(f"  FAQ instant answers:     {faq_hits}/{total} ({faq_hits/total:.0%})")
    print()

    # Per-metric breakdown
    perfect_retrieval = sum(1 for s in retrieval_scores if s == 1.0)
    high_similarity = sum(1 for s in similarity_scores if s > 0.7)
    print(f"  Perfect retrieval (100%): {perfect_retrieval}/{total}")
    print(f"  High similarity (>0.70):  {high_similarity}/{total}")
    print()

    # Grade
    overall = (avg_retrieval + avg_similarity) / 2
    if overall >= 0.85:
        grade = "A"
    elif overall >= 0.70:
        grade = "B"
    elif overall >= 0.55:
        grade = "C"
    else:
        grade = "D"
    print(f"  Overall score: {overall:.2f} — Grade: {grade}")
    print("=" * 60)


if __name__ == "__main__":
    main()
