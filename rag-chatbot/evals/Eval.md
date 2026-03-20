# Evaluation Results

## Overview

This document compares RAG pipeline evaluation results across different models, hardware configurations, and prompt strategies. All tests use the same 20-question evaluation set covering customer-facing and employee-facing queries against AutoGroup Motors documentation.

Key finding: **prompt engineering had a bigger impact on answer quality than swapping models**, and smarter models score worse on cosine similarity because they rephrase more — exposing a limitation of the metric itself.

---

## Model Comparison (Laptop — CPU Inference)

**Hardware:** ASUS Vivobook S 16 OLED — Ryzen 7 8845HS, 16 GB LPDDR5X, Radeon 780M iGPU, no discrete GPU. All models ran Q4_K_M quantization via Ollama with CPU inference.

| Model | Intelligence Index | Avg Answer Similarity | High Similarity (>0.70) | Overall Score | Grade |
|---|---|---|---|---|---|
| Gemma 2 9B | ~20 | 0.68 | 9/18 | 0.84 | B |
| Qwen3 8B (default prompt) | ~25 | 0.61 | 8/18 | 0.80 | B |
| **Qwen3 8B (optimized prompt)** | ~25 | **0.73** | **12/18** | **0.86** | **A** |
| Qwen3.5 9B (optimized prompt) | ~31 | 0.53 | 8/18 | 0.77 | B |

**Observations:**

- Retrieval precision was 100% across all model runs (18/18 perfect retrieval). The ingestion pipeline (chunking + embeddings + ChromaDB) works correctly regardless of model choice.
- Prompt optimization on Qwen3 8B produced a +0.12 jump in answer similarity — a larger gain than any model swap.
- Qwen3.5 9B, despite being the most intelligent model tested, scored the lowest on cosine similarity. Manual inspection revealed the answers were correct but heavily rephrased, which penalizes them under this metric.
- Gemma 2 9B's higher similarity score reflects its tendency to echo source material rather than paraphrase — better for cosine similarity, worse for real-world chatbot quality.

### Insight: Cosine Similarity Penalizes Smarter Models

The inverse correlation between model intelligence and similarity score is not a coincidence. More capable models rephrase, elaborate, and restructure answers. A model that parrots retrieved chunks verbatim will always score higher on surface-level word overlap metrics while being a worse conversational agent.

This finding motivates the need for an LLM-as-judge evaluation approach (see Future Work).

---

## Hardware Comparison — Qwen3.5 9B

Same model (Qwen3.5 9B, Q4_K_M, optimized prompt), same eval set, two machines.

### Laptop (CPU Inference)

**Hardware:** Ryzen 7 8845HS, 16 GB LPDDR5X, CPU-only inference (~12-15 tok/s estimated)

| Metric | Value |
|---|---|
| Questions answered | 7 / 20 |
| Questions timed out | 13 / 20 |
| Retrieval precision | 100% (on answered) |
| FAQ instant answers | 7 / 20 |

13 out of 20 questions timed out due to slow CPU inference. The model generates at ~12-15 tok/s, and longer answers (inventory lists, detailed policies) exceed the request timeout. Results are not usable for quality comparison.

### Desktop (GPU Inference)

**Hardware:** Ryzen 5 5600X, 16 GB DDR4 3600MHz, RTX 3070 8GB VRAM, CUDA inference (~55 tok/s estimated)

| Metric | Value |
|---|---|
| Questions answered | 16 / 20 |
| Questions timed out | 4 / 20 |
| Retrieval precision | 97.5% (19/20 perfect) |
| FAQ instant answers | 7 / 20 |

### Per-Question Comparison

| Q# | Topic | Laptop | Desktop | Notes |
|---|---|---|---|---|
| Q1 | Return policy | ✅ Correct | ✅ Correct | Identical quality |
| Q2 | Warranty plans | ✅ Correct | ✅ Correct | Identical quality |
| Q3 | Test drive booking | ✅ Correct | ✅ Correct | Identical quality |
| Q4 | Auto loan rates | ✅ Correct | ✅ Correct | Identical quality |
| Q5 | Synthetic oil change | ✅ Correct | ✅ Correct | Identical quality |
| Q6 | SUV stock/prices | ❌ Timeout | ❌ Timeout | List-heavy answer, needs higher timeout |
| Q7 | Referral rewards | ❌ Timeout | ⚠️ Partial | Correct concept, missing EUR amounts |
| Q8 | Electric vehicles | ❌ Timeout | ❌ Timeout | List-heavy answer |
| Q9 | Delivery to Kaunas | ❌ Timeout | ✅ Correct | Concise, missed EUR 20k waiver |
| Q10 | KASKO insurance | ❌ Timeout | ✅ Correct | Good detail, missed specific discounts |
| Q11 | Trade-in validity | ❌ Timeout | ✅ Correct | Missed 500 km condition |
| Q12 | Pre-purchase inspection | ❌ Timeout | ✅ Correct | Concise, missed OBD pricing |
| Q13 | USA import timeline | ❌ Timeout | ⚠️ Partial | Only gave timeline, missed costs/docs |
| Q14 | Brake pad cost | ❌ Timeout | ✅ Correct | Accurate price range |
| Q15 | Post-sale defect | ❌ Timeout | ❌ Timeout | Complex legal answer |
| Q16 | Salary/commission | ✅ Correct | ✅ Correct | Too broad — listed all roles |
| Q17 | Health insurance | ❌ Timeout | ❌ Timeout | Detailed coverage list |
| Q18 | Annual leave | ❌ Timeout | ✅ Correct | Excellent detail |
| Q19 | Overtime policy | ❌ Wrong | ❌ Wrong | Retrieval issue — returned salary chunk instead of overtime |
| Q20 | Employee discounts | ❌ Timeout | ⚠️ Partial | Got 3% discount, missed other benefits |

### Summary

| | Laptop (CPU) | Desktop (GPU) |
|---|---|---|
| Answered | 7 / 20 (35%) | 16 / 20 (80%) |
| Correct | 6 | 11 |
| Partial | 0 | 3 |
| Wrong | 1 | 1 |
| Timeout | 13 | 4 |
| Effective accuracy | 6/7 = 86% | 11/16 = 69% |

The desktop's lower effective accuracy is misleading — it answered 9 more questions, including harder ones that the laptop never attempted. The 4 remaining timeouts are all list-heavy or complex queries that generate long responses.

---

## Known Issues

### 1. Timeout on Long Answers
Questions requiring list-heavy responses (Q6: SUV inventory, Q8: EV list, Q15: legal process, Q17: insurance coverage) consistently time out. Fix: increase Ollama request timeout to 180s and set `num_predict: 256` to cap output length.

### 2. Q19 Retrieval/Chunking Problem
The overtime policy question consistently returns salary information instead. Both answers come from `Employee-Compensation-And-Pay-Structure.txt`, suggesting the overtime and salary content are in the same chunk or the salary chunk ranks higher. Fix: review chunk boundaries for this document or add overtime as a dedicated FAQ entry.

### 3. Cosine Similarity Metric Limitation
Surface-level word overlap does not capture semantic correctness. A model that paraphrases a correct answer will score lower than one that copies source text verbatim. This metric should be supplemented or replaced with LLM-as-judge evaluation.

---
