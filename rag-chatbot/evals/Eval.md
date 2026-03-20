# Evaluation Results

## Overview

This document tracks RAG pipeline evaluation results across 6 runs, testing different models, hardware, prompt strategies, and chunking configurations against a 20-question eval set covering customer-facing and employee-facing queries for AutoGroup Motors documentation.

**Key findings:**
- Prompt engineering had a bigger impact on answer quality than swapping models
- Smarter models score worse on cosine similarity because they rephrase more — exposing a metric limitation
- Chunk size and top_k tuning had the largest impact on answer completeness
- The best configuration so far: Qwen3.5 9B + optimized prompt + 400ch/60ov chunks + top_k=5 + keep_alive=30m

---

## Summary: All Eval Runs

| Eval | Model | Hardware | Chunk/Overlap | top_k | Answered | Timeout | Empty | Wrong | Retrieval | Notes |
|------|-------|----------|---------------|-------|----------|---------|-------|-------|-----------|-------|
| 1 | Qwen3.5 9B | Laptop (CPU) | 300/50 | 3 | 7/20 | 13 | 0 | 0 | 100% | CPU too slow, mass timeouts |
| 2 | Qwen3.5 9B | Desktop (3070) | 300/50 | 3 | 16/20 | 4 | 0 | 1 | 97.5% | Best answered count at the time |
| 3 | Qwen3.5 9B | Desktop (3070) | 300/50 | 3 | 7/20 | 0 | 12 | 1 | 97.5% | num_predict too low, empty answers |
| 4 | Qwen3.5 9B | Desktop (3070) | 1000/200 | 3 | 7/20 | 0 | 12 | 1 | 92.5% | Bigger chunks hurt retrieval precision |
| 5 | Qwen3.5 9B | Desktop (3070) | 500/75 | 5 | 13/20 | 6 | 0 | 1 | 92.5% | top_k=5 helped, but chunk size too large |
| **6** | **Qwen3.5 9B** | **Desktop (3070)** | **400/60** | **5** | **16/20** | **4** | **0** | **1** | **92.5%** | **Best overall — keep_alive=30m, highest answer quality** |

---

## Eval 6: Best Run (Current Configuration)

**Config:** Qwen3.5 9B Q4_K_M · RTX 3070 · 400ch/60ov · top_k=5 · keep_alive=30m · no num_predict cap

| Q# | Topic | Type | Result | Notes |
|----|-------|------|--------|-------|
| Q1 | Return policy | FAQ/Customer | ✅ Correct | Added in-person policy as bonus context |
| Q2 | Warranty plans | FAQ/Customer | ✅ Correct | All prices correct, missing km limits |
| Q3 | Test drive booking | FAQ/Customer | ✅ Correct | Added insurance excess detail |
| Q4 | Auto loan rates | FAQ/Customer | ✅ Correct | All key numbers present |
| Q5 | Synthetic oil change | FAQ/Customer | ✅ Correct | Perfect answer |
| Q6 | SUV stock/prices | Customer | ❌ Timeout | List-heavy, needs long generation |
| Q7 | Referral rewards | Customer | ✅ Correct | EUR amounts included, excellent detail |
| Q8 | Electric vehicles | Customer | ❌ Timeout | List-heavy inventory question |
| Q9 | Delivery to Kaunas | Customer | ✅ Correct | Included EUR 20k waiver this time |
| Q10 | KASKO insurance | Customer | ✅ Correct | Detailed coverage, specific discounts |
| Q11 | Trade-in validity | Customer | ✅ Correct | Both 7-day and 500km conditions |
| Q12 | Pre-purchase inspection | Customer | ❌ Timeout | 0% retrieval — correct doc not in results |
| Q13 | USA import timeline | Customer | ⚠️ Partial | Timeline correct, missing costs/docs |
| Q14 | Brake pad cost | Customer | ✅ Correct | Price + VAT + warranty info |
| Q15 | Post-sale defect | Customer | ❌ Timeout | Complex legal answer |
| Q16 | Salary/commission | FAQ/Employee | ✅ Correct | Broad but accurate |
| Q17 | Health insurance | Employee | ✅ Correct | **Excellent** — all coverage amounts listed |
| Q18 | Annual leave | Employee | ✅ Correct | All tiers and carryover rules |
| Q19 | Overtime policy | FAQ/Employee | ❌ Wrong | Returns salary info — retrieval/chunking bug |
| Q20 | Employee discounts | Employee | ❌ Timeout | Retrieval 50%, wrong docs ranked higher |

**Answered: 16/20 · Correct: 13 · Partial: 1 · Wrong: 1 · Timeout: 4**

### Progress from Eval 2 to Eval 6

Key improvements between the two 16/20 runs:

| Metric | Eval 2 (300/50, top_k=3) | Eval 6 (400/60, top_k=5) |
|--------|--------------------------|--------------------------|
| Q7 (Referral) | ⚠️ Missing EUR amounts | ✅ Full amounts included |
| Q9 (Delivery) | ⚠️ Missing EUR 20k waiver | ✅ Waiver mentioned |
| Q10 (KASKO) | ✅ Good | ✅ Better — specific discounts |
| Q11 (Trade-in) | ⚠️ Missing 500km condition | ✅ Both conditions |
| Q14 (Brake pads) | ✅ Basic | ✅ Added warranty info |
| Q17 (Health insurance) | ❌ Timeout | ✅ Full coverage breakdown |
| Q18 (Annual leave) | ✅ Good | ✅ Good |

The higher top_k directly improved answer completeness — the model gets more context chunks and can synthesize more detailed answers.

---

## Model Comparison (Laptop — CPU Inference)

**Hardware:** ASUS Vivobook S 16 OLED — Ryzen 7 8845HS, 16 GB LPDDR5X, Radeon 780M iGPU. All models Q4_K_M via Ollama, CPU inference, 300ch/50ov chunks.

| Model | Intelligence Index | Avg Similarity | High Similarity (>0.70) | Overall Score | Grade |
|-------|-------------------|----------------|------------------------|---------------|-------|
| Gemma 2 9B | ~20 | 0.68 | 9/18 | 0.84 | B |
| Qwen3 8B (default prompt) | ~25 | 0.61 | 8/18 | 0.80 | B |
| **Qwen3 8B (optimized prompt)** | ~25 | **0.73** | **12/18** | **0.86** | **A** |
| Qwen3.5 9B (optimized prompt) | ~31 | 0.53 | 8/18 | 0.77 | B |

### Insight: Cosine Similarity Penalizes Smarter Models

There is a clear inverse correlation between model intelligence and similarity score. More capable models rephrase, elaborate, and restructure answers rather than echoing source chunks. Gemma 2 9B scores highest because it parrots chunks verbatim — good for cosine similarity, bad for real chatbot quality.

Prompt optimization on Qwen3 8B produced a +0.12 jump in answer similarity — a larger improvement than any model swap. This demonstrates that prompt engineering matters more than model selection at this scale.

---

## Configuration Experiments

### Chunk Size Impact

| Config | Retrieval Precision | Answer Completeness | Verdict |
|--------|-------------------|-------------------|---------|
| 300ch / 50ov | 97.5% | Lower — small chunks lack full context | Best retrieval precision |
| 400ch / 60ov | 92.5% | Higher — enough context per chunk | Best balance |
| 500ch / 75ov | 92.5% | Similar to 400 | No improvement over 400 |
| 1000ch / 200ov | 92.5% | Worse — too much noise per chunk | Too large |

### top_k Impact

| top_k | Context Size (400ch) | Effect |
|-------|---------------------|--------|
| 3 | ~1200 chars | Misses info split across chunks |
| 5 | ~2000 chars | Better completeness, slight retrieval dilution |

---

## Known Issues

### 1. Persistent Timeouts (Q6, Q8, Q15, Q20)
Questions needing long list-heavy answers still timeout. Q6 (list all SUVs with prices) and Q8 (list all EVs with ranges) require generating 200+ tokens of structured data. Possible fixes: increase Ollama timeout further, or pre-cache FAQ answers for inventory-type questions.

### 2. Q12 Retrieval Failure
Pre-purchase inspection pricing (in `Service-And-Maintenance-Price-List.txt`) fails to appear in top-5 results with 400ch chunks. The term "pre-purchase inspection" may not embed closely enough to the chunk containing the pricing table. Fix: ensure this pricing info exists in a chunk with clear semantic markers, or add a dedicated FAQ entry.

### 3. Q19 Wrong Answer — Chunking Bug
Overtime policy question consistently returns salary information. Both topics exist in `Employee-Compensation-And-Pay-Structure.txt`. The salary chunk ranks higher than the overtime chunk because salary terms are more semantically similar to "pay rates" in the question. Fix: split overtime into a separate document section or add as an FAQ entry.

### 4. Cosine Similarity Metric Limitation
Surface-level word overlap does not measure semantic correctness. This eval should be supplemented with LLM-as-judge scoring to measure whether answers are actually correct regardless of phrasing.

---

## Future Work

- **LLM-as-judge evaluation**: Send (question, expected, actual) triples to an LLM API to rate correctness 1-5, replacing cosine similarity as primary metric
- **Hybrid retrieval (BM25 + vector)**: Add keyword matching to complement semantic search — would help Q12 and Q19 where exact terms matter
- **Re-ranking with cross-encoder**: Re-rank top-k results with `cross-encoder/ms-marco-MiniLM-L-6-v2` before passing to LLM
- **PDF ingestion support**: Extend document loader beyond .txt and .md
- **Dockerize**: `docker-compose.yml` for one-command setup with Ollama + FastAPI