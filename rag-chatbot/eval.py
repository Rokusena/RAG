"""
eval.py — Evaluation script for the RAG chatbot.

Runs all 20 questions, compares expected vs actual answers,
and saves a full report to evals/eval_<nr>.txt.

Usage:
    python eval.py
"""

import os
import sys
from datetime import datetime

# Add parent directory context
sys.path.insert(0, os.path.dirname(__file__))

from query import get_retriever, answer_question, _match_faq, OLLAMA_MODEL

EVALS_DIR = os.path.join(os.path.dirname(__file__), "evals")

# --- Evaluation Dataset (20 questions) ---
# Each entry: question, expected answer (key facts), expected source files, mode

EVAL_DATASET = [
    # ── Customer questions (15) ──
    {
        "question": "What is the return policy for vehicles bought online?",
        "expected_answer": "Distance/online purchases have a 14-day cooling-off period from delivery date. The vehicle must not be driven more than 300 km beyond delivery odometer, must be in original condition, and all documentation/keys/accessories returned. Refunds are processed within 14 business days after inspection.",
        "expected_sources": ["Returns-Refunds-And-Dispute-Resolution.txt"],
        "mode": "customer",
    },
    {
        "question": "What are the extended warranty plans and their prices?",
        "expected_answer": "Bronze Plan: powertrain only, 12 months or 20,000 km, EUR 299. Silver Plan: powertrain plus electrical, AC, steering, 18 months or 30,000 km, EUR 549. Gold Plan: bumper-to-bumper (excludes wear items), 24 months or 40,000 km, EUR 899. All plans include 24/7 roadside assistance within Lithuania and Baltics.",
        "expected_sources": ["Warranty-And-After-Sales-Policy.txt"],
        "mode": "customer",
    },
    {
        "question": "How can I book a test drive and what do I need?",
        "expected_answer": "Book online at www.autogroupmotors.lt/test-drive, by phone at +370 5 123 4567, or walk in. Available Mon-Fri 9:00-17:00 and Sat 9:00-14:00. Each slot is 30 minutes (15-20 min driving). You need a valid driving license (Category B) and a valid ID. Must be 18+ and not impaired.",
        "expected_sources": ["Test-Drive-Policy-And-Procedure.txt"],
        "mode": "customer",
    },
    {
        "question": "What auto loan interest rates do you offer?",
        "expected_answer": "Consumer auto loan rates start from 4.9% APR with strong credit, through partner banks SEB, Swedbank, Luminor, and Šiaulių Bankas. Loan terms 12-84 months, minimum 10% down payment. Pre-approval within 2 business hours. First-Time Buyer Program (ages 18-25) at 6.5% APR for vehicles up to EUR 15,000 with 20% down or co-signer.",
        "expected_sources": ["Financing-And-Leasing-Options.txt"],
        "mode": "customer",
    },
    {
        "question": "How much does a synthetic oil change cost?",
        "expected_answer": "Synthetic oil change costs EUR 79. Includes up to 5-6 liters of oil and oil filter. Prices include 21% VAT. Conventional oil is EUR 49 and diesel vehicles EUR 89.",
        "expected_sources": ["Service-And-Maintenance-Price-List.txt"],
        "mode": "customer",
    },
    {
        "question": "What SUVs do you currently have in stock and at what prices?",
        "expected_answer": "Current SUV stock includes: Volvo XC60 EUR 42,500, BMW X3 EUR 39,900, Mercedes-Benz GLC EUR 37,800, Toyota RAV4 EUR 34,200, Hyundai Tucson EUR 33,900, Volkswagen Tiguan EUR 27,500, Mazda CX-5 EUR 28,700, Land Rover Discovery Sport EUR 26,900.",
        "expected_sources": ["Vehicle-Stock-And-Inventory.txt"],
        "mode": "customer",
    },
    {
        "question": "What do I get for referring a friend?",
        "expected_answer": "When you refer a new customer who completes a purchase, you receive EUR 200 service credit or EUR 150 cash. The referred customer gets EUR 100 service credit. There is no limit on referrals. Rewards are processed within 14 business days.",
        "expected_sources": ["Customer-Loyalty-And-Referral-Program.txt"],
        "mode": "customer",
    },
    {
        "question": "What electric vehicles do you sell and what is their range?",
        "expected_answer": "Tesla Model 3 Long Range EUR 39,900 (602 km range), Volkswagen ID.4 EUR 34,500 (520 km range), BMW iX1 EUR 44,200 (440 km range), Volvo XC40 Recharge EUR 37,800, Mercedes EQA 250 EUR 33,500.",
        "expected_sources": ["Vehicle-Stock-And-Inventory.txt"],
        "mode": "customer",
    },
    {
        "question": "How much does delivery to Kaunas cost and how long does it take?",
        "expected_answer": "Delivery to Kaunas costs EUR 79 and takes 1-2 business days. Delivery fee is waived for purchases over EUR 20,000. Local delivery within Vilnius area (up to 30 km) is complimentary.",
        "expected_sources": ["Vehicle-Delivery-And-Shipping-Options.txt"],
        "mode": "customer",
    },
    {
        "question": "What KASKO insurance options are available and how much do they cost?",
        "expected_answer": "KASKO (comprehensive) insurance costs 2-6% of insured vehicle value annually (e.g., EUR 300-EUR 900 for a EUR 15,000 vehicle). Partners include ERGO, Gjensidige, BTA, and Compensa Vienna Insurance Group. Gjensidige offers 10% discount for dashcams or ADAS systems. BTA offers Mini-KASKO covering theft and total loss at reduced rates. Complimentary claims assistance for first 12 months.",
        "expected_sources": ["Insurance-Options-And-Partners.txt"],
        "mode": "customer",
    },
    {
        "question": "How long is a trade-in offer valid for?",
        "expected_answer": "Trade-in offers are valid for 7 calendar days or 500 km driven (whichever comes first). The in-person appraisal inspection takes 20-30 minutes. Popular brands like VW, Toyota, BMW, Audi tend to hold value better. SUVs/crossovers command a premium.",
        "expected_sources": ["Trade-In-And-Vehicle-Appraisal-Guide.txt"],
        "mode": "customer",
    },
    {
        "question": "How much does a pre-purchase vehicle inspection cost?",
        "expected_answer": "A pre-purchase inspection costs EUR 89. OBD-II diagnostic scan is EUR 29 (waived if repair is done). Advanced diagnostics cost EUR 59. All prices include 21% VAT.",
        "expected_sources": ["Service-And-Maintenance-Price-List.txt"],
        "mode": "customer",
    },
    {
        "question": "How long does it take to import and register a car from the USA?",
        "expected_answer": "Importing from the USA takes 4-6 weeks for registration. Non-EU customs duty is 6.5% on customs value (CIF), plus 21% VAT. A 150-point inspection is performed upon arrival. Registration assistance fee is EUR 150 (included for customer purchases). Required documents include original title, customs declaration (SAD), VAT proof, technical inspection certificate, insurance, and ID.",
        "expected_sources": ["Vehicle-Import-And-Registration-Process.txt"],
        "mode": "customer",
    },
    {
        "question": "How much does front brake pad replacement cost?",
        "expected_answer": "Front brake pad replacement costs EUR 89-EUR 149 depending on the vehicle. Rear brake pads cost EUR 79-EUR 139. Brake disc replacement is EUR 159-EUR 249. Brake fluid flush is EUR 49. All prices include 21% VAT. Service work carries a 6-month or 10,000 km warranty on parts and labor.",
        "expected_sources": ["Service-And-Maintenance-Price-List.txt"],
        "mode": "customer",
    },
    {
        "question": "What happens if I discover a defect after buying a vehicle?",
        "expected_answer": "If a defect is discovered within 6 months that existed at time of sale, the burden of proof lies with AutoGroup Motors to show it wasn't pre-existing. Options include free repair, replacement vehicle, partial refund, or full refund. Inspection is scheduled within 5 business days. Disputes can be escalated to internal review (5 business days), Lithuanian Consumer Rights Authority, EU ODR platform, or court.",
        "expected_sources": ["Returns-Refunds-And-Dispute-Resolution.txt"],
        "mode": "customer",
    },
    # ── Employee questions (5) ──
    {
        "question": "What is the salary range for a sales consultant and how does commission work?",
        "expected_answer": "Sales Consultant base salary is EUR 1,600-2,200 gross per month. Commission is 1.5% of vehicle gross profit at target margin and 0.75% below target. Add-on commission is 5%. Monthly volume bonus of EUR 200 for every 5 vehicles sold beyond the 8-vehicle monthly target. Annual bonus up to 1.5 months' salary if targets exceeded.",
        "expected_sources": ["Employee-Compensation-And-Pay-Structure.txt"],
        "mode": "employee",
    },
    {
        "question": "What does the private health insurance cover?",
        "expected_answer": "Private insurance through Compensa Vienna covers: outpatient care EUR 2,000/year, dental EUR 500/year, inpatient EUR 15,000/year, medications 80% reimbursement up to EUR 400/year, mental health 12 sessions/year fully covered, optical EUR 150/year. Family members can be added for EUR 45/person/month. Eligible after probation period for full-time employees.",
        "expected_sources": ["Employee-Health-And-Benefits-Package.txt"],
        "mode": "employee",
    },
    {
        "question": "How many annual leave days do employees get?",
        "expected_answer": "Standard annual leave is 20 working days for full-time employees. Employees with 5+ years get 25 days, and 10+ years get 28 days. Maximum 5 days can be carried over to the next year. Sick leave is 80% of salary for the first 2 days (paid by employer), then 62.06% from SODRA.",
        "expected_sources": ["Employee-Health-And-Benefits-Package.txt"],
        "mode": "employee",
    },
    {
        "question": "What is the overtime policy and pay rates?",
        "expected_answer": "Overtime must be pre-approved by your manager. Rates: weekday overtime 1.5x hourly rate, Saturday 1.5x, Sunday and public holidays 2.0x. Maximum 8 hours overtime per week and 180 hours per calendar year per Lithuanian labour law.",
        "expected_sources": ["Employee-Compensation-And-Pay-Structure.txt"],
        "mode": "employee",
    },
    {
        "question": "What employee discounts and benefits are available for vehicle purchases?",
        "expected_answer": "Employees get a 3% discount off listed vehicle price (once per year) per the Health and Benefits package. The Employee Handbook also mentions a 10% vehicle purchase discount. Additional benefits include EUR 100/month fuel allowance (if approved), EUR 500/year professional development budget, and EUR 50/month voluntary pension match through SEB bank.",
        "expected_sources": ["Employee-Health-And-Benefits-Package.txt", "Employee-Handbook-And-Code-Of-Conduct.txt"],
        "mode": "employee",
    },
]


def evaluate_retrieval_precision(actual_sources: list[str], expected_sources: list[str]) -> float:
    """What fraction of expected sources appear in retrieved sources."""
    if not expected_sources:
        return 1.0
    hits = sum(1 for s in expected_sources if s in actual_sources)
    return hits / len(expected_sources)


def get_next_eval_number() -> int:
    """Find the next eval number by scanning existing files in evals/."""
    os.makedirs(EVALS_DIR, exist_ok=True)
    existing = [f for f in os.listdir(EVALS_DIR) if f.startswith("eval_") and f.endswith(".txt")]
    if not existing:
        return 1
    numbers = []
    for f in existing:
        try:
            num = int(f.replace("eval_", "").replace(".txt", ""))
            numbers.append(num)
        except ValueError:
            pass
    return max(numbers, default=0) + 1


def main():
    eval_nr = get_next_eval_number()

    print("=" * 70)
    print(f"  RAG Chatbot Evaluation #{eval_nr}")
    print("=" * 70)

    print("\nLoading models and collections...")
    collections, model = get_retriever()

    total = len(EVAL_DATASET)
    retrieval_scores = []
    faq_hits = 0
    results = []

    print(f"Running {total} evaluation queries...\n")

    for i, entry in enumerate(EVAL_DATASET, 1):
        question = entry["question"]
        expected = entry["expected_answer"]
        expected_src = entry["expected_sources"]
        mode = entry["mode"]

        # Check if FAQ handles this
        faq = _match_faq(question, mode)
        is_faq = faq is not None
        if is_faq:
            faq_hits += 1

        # Get actual answer
        result = answer_question(question, collections, model, mode=mode)
        actual_answer = result["answer"]
        actual_sources = result["sources"]

        # Retrieval precision
        precision = evaluate_retrieval_precision(actual_sources, expected_src)
        retrieval_scores.append(precision)

        faq_tag = " [FAQ]" if is_faq else ""
        status = "OK" if precision == 1.0 else "MISS"
        print(f"  [{i:2d}/{total}] {status}{faq_tag} | Retrieval: {precision:.0%} | {question[:50]}")

        results.append({
            "nr": i,
            "question": question,
            "mode": mode,
            "is_faq": is_faq,
            "expected_answer": expected,
            "actual_answer": actual_answer,
            "expected_sources": expected_src,
            "actual_sources": actual_sources,
            "retrieval": precision,
        })

    # ── Summary ──
    avg_retrieval = sum(retrieval_scores) / len(retrieval_scores) if retrieval_scores else 0
    perfect_retrieval = sum(1 for s in retrieval_scores if s == 1.0)

    print()
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Total questions:         {total}")
    print(f"  Avg retrieval precision: {avg_retrieval:.1%}")
    print(f"  Perfect retrieval:       {perfect_retrieval}/{total}")
    print(f"  FAQ instant answers:     {faq_hits}/{total}")
    print("=" * 70)

    # ── Write report file ──
    report_path = os.path.join(EVALS_DIR, f"eval_{eval_nr}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{'=' * 70}\n")
        f.write(f"  RAG Chatbot Evaluation #{eval_nr}\n")
        f.write(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"  Model: {OLLAMA_MODEL}\n")
        f.write(f"  Questions: {total}\n")
        f.write(f"{'=' * 70}\n\n")

        for r in results:
            faq_tag = " [FAQ]" if r["is_faq"] else ""
            f.write(f"{'-' * 70}\n")
            f.write(f"  Q{r['nr']}/{total}{faq_tag} ({r['mode'].upper()}) | Retrieval: {r['retrieval']:.0%}\n")
            f.write(f"{'-' * 70}\n\n")
            f.write(f"  Question:\n")
            f.write(f"    {r['question']}\n\n")
            f.write(f"  Expected answer:\n")
            f.write(f"    {r['expected_answer']}\n\n")
            f.write(f"  Actual answer:\n")
            f.write(f"    {r['actual_answer']}\n\n")
            f.write(f"  Sources expected: {r['expected_sources']}\n")
            f.write(f"  Sources actual:   {r['actual_sources']}\n\n")

        f.write(f"{'=' * 70}\n")
        f.write(f"  RESULTS SUMMARY\n")
        f.write(f"{'=' * 70}\n")
        f.write(f"  Total questions:         {total}\n")
        f.write(f"  Avg retrieval precision: {avg_retrieval:.1%}\n")
        f.write(f"  Perfect retrieval:       {perfect_retrieval}/{total}\n")
        f.write(f"  FAQ instant answers:     {faq_hits}/{total}\n")
        f.write(f"{'=' * 70}\n")

    print(f"\n  Report saved to: {report_path}")
    print()


if __name__ == "__main__":
    main()
