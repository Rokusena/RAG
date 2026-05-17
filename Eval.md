# Evaluation Results

## Overview

This document tracks RAG pipeline evaluation results across 9 runs, testing different models, hardware, prompt strategies, chunking configurations, and **embedding providers** against a 20-question eval set covering customer-facing and employee-facing queries for AutoGroup Motors documentation.

**Key findings:**
- Prompt engineering had a bigger impact on answer quality than swapping models
- Smarter models score worse on cosine similarity because they rephrase more — exposing a metric limitation
- Chunk size and top_k tuning had a large impact on answer completeness
- **Switching embeddings from `all-MiniLM-L6-v2` to OpenAI `text-embedding-3-small` lifted retrieval precision from 92.5% to 100%** (Eval 9)
- **Switching the LLM from Qwen3.5 9B (local) to OpenAI gpt-4o-mini eliminated all timeouts** (Eval 9)
- **Atomic chunking for the inventory document fixed the only hallucination in Eval 9** — Q6 SUV stock now lists all 8 vehicles correctly (Eval 10)
- **Prompt enforcement of "completeness" is one-sided**: gpt-4o-mini honors "list all items" but does not proactively add secondary facts (VAT lines, fee waivers, warranty notes) — answers got *shorter*, not more complete (Eval 10)
- **Eval 11 chunk-level diagnostic proves the partials are mostly prompt-driven, not retrieval-driven**: 9 of 11 partials have facts present in the retrieved context that the model simply omits. File-level retrieval precision (100%) is no longer informative.
- The current best configuration: gpt-4o-mini + text-embedding-3-small + 400ch/60ov chunks + atomic inventory + top_k=5

---

## Summary: All Eval Runs

| Eval | LLM | Embeddings | Chunk/Overlap | top_k | Answered | Timeout | Empty | Wrong | Retrieval | Notes |
|------|-----|------------|---------------|-------|----------|---------|-------|-------|-----------|-------|
| 1 | Qwen3.5 9B (CPU laptop) | MiniLM-L6 | 300/50 | 3 | 7/20 | 13 | 0 | 0 | 100% | CPU too slow, mass timeouts |
| 2 | Qwen3.5 9B (3070) | MiniLM-L6 | 300/50 | 3 | 16/20 | 4 | 0 | 1 | 97.5% | Best answered at the time |
| 3 | Qwen3.5 9B (3070) | MiniLM-L6 | 300/50 | 3 | 7/20 | 0 | 12 | 1 | 97.5% | num_predict too low, empty answers |
| 4 | Qwen3.5 9B (3070) | MiniLM-L6 | 1000/200 | 3 | 7/20 | 0 | 12 | 1 | 92.5% | Bigger chunks hurt retrieval precision |
| 5 | Qwen3.5 9B (3070) | MiniLM-L6 | 500/75 | 5 | 13/20 | 6 | 0 | 1 | 92.5% | top_k=5 helped, chunks too large |
| 6 | Qwen3.5 9B (3070) | MiniLM-L6 | 400/60 | 5 | 16/20 | 4 | 0 | 1 | 92.5% | Best local-only run |
| 9 | gpt-4o-mini (OpenAI) | text-embedding-3-small | 400/60 | 5 | 20/20 | 0 | 0 | 1 | 100% | First OpenAI run — terseness issue identified |
| **10** | **gpt-4o-mini (OpenAI)** | **text-embedding-3-small** | **400/60 + atomic** | **5** | **20/20** | **0** | **0** | **0** | **100%** | **Best overall — atomic inventory + tightened prompt** |
| 11 | gpt-4o-mini (OpenAI) | text-embedding-3-small | 400/60 + atomic | 5 | 20/20 | 0 | 0 | 0 | 100% | Same config as Eval 10 — added chunk-level diagnostic to eval reports |

---

## Eval 11: Chunk-Level Diagnostic (Same Config as Eval 10)

**Config:** Identical to Eval 10. The only change was adding the actual retrieved chunks (with source, distance, and full text) to each question's eval report — see [eval.py](rag-chatbot/eval.py) and [query.py](rag-chatbot/query.py:119).

The metrics are unchanged (20/20 answered, 0 wrong, 0 timeouts, 100% file-level retrieval). The value of Eval 11 is in **knowing why partials are partial** for the first time.

### Diagnosis of all 11 partial / mixed questions

For each missing fact in each currently-partial answer, we now know whether the fact was in the retrieved context or not:

| Q | Missing fact | Was it in retrieved chunks? | Failure type |
|---|---|---|---|
| Q8 | All 5 EV prices | ✅ Chunk 4 (full inventory) | Prompt — model used ranges, ignored prices |
| Q9 | EUR 20k delivery waiver | ❌ Not retrieved | Retrieval |
| Q9 | Vilnius 30km free zone | ✅ Chunk 4 | Prompt |
| Q10 | ERGO / Gjensidige / BTA partner names | ✅ Chunk 4 | Prompt |
| Q10 | Compensa, Gjensidige 10% ADAS discount | ❌ Not retrieved | Retrieval |
| Q11 | 20-30 min appraisal, brand value info | ❌ Not retrieved | Retrieval |
| Q12 | OBD EUR 29, Advanced EUR 59 | ✅ Chunk 1 | Prompt |
| Q12 | 21% VAT line | ❌ Not retrieved | Retrieval |
| Q13 | Customs duty 6.5%, 21% VAT, document list | ❌ Not retrieved | Retrieval |
| Q14 | Rear / disc / fluid prices, 21% VAT, 6-month warranty | ✅ Chunks 1 & 5 | **Pure terseness** |
| Q15 | Partial/full refund options, ODR/court escalation | ❌ Not retrieved | Retrieval |
| Q15 | Test-Drive content leaked into answer | — | Prompt (low-relevance chunk used) |
| Q17 | Family add-on EUR 45 / person / month | ✅ Chunk 2 | Prompt |
| Q18 | Sick leave 80% / SODRA 62.06% | ✅ Chunk 2 | Prompt |
| Q20 | EUR 50/mo SEB pension match | ❌ Not retrieved | Retrieval |
| Q20 | Customer-Loyalty doc contaminated answer | — | Cross-collection leak |

### Summary by failure type

| Category | Count | Questions |
|---|---|---|
| **Pure prompt** (facts in chunks, model omits them) | 4 | Q8, Q14, Q17, Q18 |
| **Pure retrieval** (relevant chunks not in top_k) | 2 | Q11, Q13 |
| **Mixed** (prompt + retrieval) | 5 | Q9, Q10, Q12, Q15, Q20 |
| **Cross-collection leak** | 1 | Q20 (Customer-Loyalty doc retrieved in employee mode) |

**9 of 11 partials have a prompt component.** The model is ignoring facts that are sitting in the retrieved context. Tuning chunk size or top_k cannot fix this dominant pattern — only prompt engineering can.

**7 of 11 partials also have a retrieval gap.** Bumping `TOP_K` from 5 → 8 would likely close most of these (Q9 waiver, Q11 appraisal/brand, Q13 customs/docs, Q15 refunds/escalation, Q20 pension).

### Distance observations

- **The good:** Q9 chunk 1 distance 0.5163, Q7 chunk 1 distance 0.8203, Q14 chunk 1 distance 0.7195 — `text-embedding-3-small` ranks the right chunks first.
- **The trap:** Q15 chunk 5 distance 0.8074 is a Test-Drive chunk that polluted the answer. Low distance ≠ topical relevance — the chunk is about post-test-drive procedure, which shares vocabulary with "after buying."
- **The cross-collection leak:** Q20 chunk 3 distance 0.9050 is a Customer-Loyalty chunk retrieved into the *employee* collection. This means at ingest time, the Customer-Loyalty doc *was* added to the employee collection (employees see everything per [ingest.py:120](rag-chatbot/ingest.py#L120)). The leak is a content-design issue: the Customer-Loyalty doc shares the words "discount," "Gold," "tier" with the employee benefits doc.

---

## Eval 10: Best Run (Current Configuration)

**Config:** gpt-4o-mini · `text-embedding-3-small` (1536-dim) · 400ch/60ov · **atomic chunking for `Vehicle-Stock-And-Inventory.txt`** · tightened OpenAI `SYSTEM_PROMPT` · top_k=5

### Eval 9 vs Eval 10 — per-question diff

| Q# | Topic | Type | Eval 9 | Eval 10 | Δ |
|----|-------|------|--------|---------|---|
| Q1 | Return policy | FAQ | ✅ | ✅ | = |
| Q2 | Warranty plans | FAQ | ⚠️ | ⚠️ | = (FAQ entry unchanged) |
| Q3 | Test drive | FAQ | ✅ | ✅ | = |
| Q4 | Auto loan rates | FAQ | ✅ | ✅ | = |
| Q5 | Oil change | FAQ | ✅ | ✅ | = |
| Q6 | **SUV stock** | LLM | ❌ Wrong | ✅ All 8 listed with prices | **↑↑ huge** |
| Q7 | Referral | LLM | ✅ | ✅ (closing pleasantry removed) | = |
| Q8 | **EVs** | LLM | ⚠️ 4 models, no prices | ⚠️ 5 models + EQA added, **still no prices** | ↑ partial |
| Q9 | Delivery to Kaunas | LLM | ⚠️ | ⚠️ Even terser — one sentence | ↓ |
| Q10 | KASKO | LLM | ⚠️ | ⚠️ Added 5% bundle discount, partner names still missing | ≈ |
| Q11 | Trade-in | LLM | ⚠️ | ⚠️ Even terser — one sentence | ↓ |
| Q12 | Pre-purchase inspection | LLM | ⚠️ EUR 89 + extra | ⚠️ Only "EUR 89." | ↓ |
| Q13 | USA import | LLM | ⚠️ | ⚠️ | ≈ |
| Q14 | Brake pads | LLM | ⚠️ Front only | ⚠️ Front only, terser | ↓ |
| Q15 | Defect handling | LLM | ⚠️ | ⚠️ More detail but contaminated with Test-Drive content | ≈ |
| Q16 | Salary | FAQ | ⚠️ | ⚠️ | = |
| Q17 | Health insurance | LLM | ✅ | ✅ (lost family add-on line) | ≈ |
| Q18 | Annual leave | LLM | ⚠️ | ⚠️ | = |
| Q19 | Overtime | FAQ | ✅ | ✅ | = |
| Q20 | Employee discounts | LLM | ⚠️ Conflated tiers | ⚠️ Both 3% and 10% mentioned, still confused | ↑ slight |

**Eval 10 totals:** ✅ 8 correct · ⚠️ 12 partial · ❌ 0 wrong · 0 timeouts · 100% retrieval
**Eval 9 totals:** ✅ 7 correct · ⚠️ 12 partial · ❌ 1 wrong · 0 timeouts · 100% retrieval

### What changed between Eval 9 and Eval 10

| Dimension | Eval 9 | Eval 10 | Effect |
|---|---|---|---|
| Chunking | RecursiveCharacterTextSplitter on all files (400/60) | Same, **except `Vehicle-Stock-And-Inventory.txt` stored as one 6391-char chunk** | Q6 fully fixed; Q8 picked up Mercedes EQA |
| `SYSTEM_PROMPT` | 8 numbered rules, "concise but complete" | Rewrote with explicit "COMPLETENESS" top rule, anti-closing-pleasantry list, enumerate-when-asked-to-list | Closing pleasantries eliminated; Q6 hallucination eliminated; **but single-fact answers got shorter, not more complete** |
| New code | — | `ATOMIC_FILES` set in [config.py](rag-chatbot/config.py); branch in `chunk_documents()` in [ingest.py](rag-chatbot/ingest.py) | Cleaner ingest path for list-heavy documents |

### Verdict on the prompt change

The new prompt produced **asymmetric** behavior:
- **Wins:** enumeration when explicitly asked to list (Q6, Q8 EVs now enumerated); zero "feel free to reach out!" closings; no hallucinations.
- **Losses:** gpt-4o-mini interpreted "completeness" as "answer the literal question" — so single-fact prompts like "how much is X?" got *terser* (Q9, Q11, Q12, Q14). It dropped warranty notes, VAT lines, secondary prices, fee waivers, and free-zone caveats that Eval 9 had included.

The "no closing pleasantries" rule appears to have been generalized by the model into "do not append anything beyond the literal answer," which strips related context the user benefits from seeing.

### Fix path forward

Adjust [config.py](rag-chatbot/config.py) `SYSTEM_PROMPT` to explicitly require **related secondary facts** even on single-fact questions:
- Add an instruction like: *"If the context contains related numeric facts (other tier prices, VAT lines, free-zone exceptions, warranty windows, secondary fees), include them after the primary answer — even if the user only asked about one."*
- Or, more robust: add 1-2 few-shot examples in the prompt showing a short question producing a complete multi-fact answer.

---

## Eval 9: Previous Best — Identified the Terseness Issue

**Config:** OpenAI gpt-4o-mini · OpenAI `text-embedding-3-small` (1536-dim) · 400ch/60ov · top_k=5 · original OpenAI prompt

| Q# | Topic | Type | Result | Notes |
|----|-------|------|--------|-------|
| Q1 | Return policy | FAQ/Customer | ✅ Correct | FAQ — adds in-person policy as bonus |
| Q2 | Warranty plans | FAQ/Customer | ⚠️ Partial | FAQ — missing km limits & Baltic roadside |
| Q3 | Test drive booking | FAQ/Customer | ✅ Correct | All key info |
| Q4 | Auto loan rates | FAQ/Customer | ✅ Correct | All numbers present |
| Q5 | Synthetic oil change | FAQ/Customer | ✅ Correct | Perfect |
| Q6 | SUV stock/prices | Customer | ❌ Wrong | Hallucinated Kia Sportage, omitted 4 models, no prices |
| Q7 | Referral rewards | Customer | ✅ Correct | EUR amounts included |
| Q8 | Electric vehicles | Customer | ⚠️ Partial | All prices missing, Mercedes EQA absent |
| Q9 | Delivery to Kaunas | Customer | ⚠️ Partial | Missing EUR 20k waiver & Vilnius free zone |
| Q10 | KASKO insurance | Customer | ⚠️ Partial | Partner names missing, fabricated tier names |
| Q11 | Trade-in validity | Customer | ⚠️ Partial | Missing appraisal & brand info |
| Q12 | Pre-purchase inspection | Customer | ⚠️ Partial | Only EUR 89 — no OBD/advanced/VAT detail |
| Q13 | USA import timeline | Customer | ⚠️ Partial | Timeline only — no customs, VAT, docs |
| Q14 | Brake pad cost | Customer | ⚠️ Partial | Front only — no rear/disc/fluid/VAT/warranty |
| Q15 | Post-sale defect | Customer | ⚠️ Partial | Repair/replacement only — no refunds, escalation |
| Q16 | Salary/commission | FAQ/Employee | ⚠️ Partial | FAQ — missing 0.75% below-target, add-on, volume bonus |
| Q17 | Health insurance | Employee | ✅ Correct | Excellent — full breakdown |
| Q18 | Annual leave | Employee | ⚠️ Partial | Missing sick leave detail |
| Q19 | Overtime policy | FAQ/Employee | ✅ Correct | Perfect FAQ match |
| Q20 | Employee discounts | Employee | ⚠️ Partial | Conflated loyalty tiers with employee perks |

**Answered: 20/20 · Correct: 7 · Partial: 12 · Wrong: 1 · Timeouts: 0 · Retrieval: 100%**

### What changed between Eval 6 and Eval 9

| Dimension | Eval 6 | Eval 9 | Effect |
|---|---|---|---|
| LLM | Qwen3.5 9B (local, RTX 3070) | OpenAI gpt-4o-mini | **0 timeouts** (was 4); answered 20/20 (was 16/20) |
| Embeddings | `all-MiniLM-L6-v2` (384-dim, EN-only) | `text-embedding-3-small` (1536-dim, multilingual) | **Retrieval 100%** (was 92.5%); Q12 retrieval now resolves |
| Chunking | 400/60 · top_k=5 | 400/60 · top_k=5 | Unchanged |
| Prompt | `SYSTEM_PROMPT_OLLAMA_*` | `SYSTEM_PROMPT` (OpenAI variant) | Different prompt active — see new failure mode below |

### New failure mode: answer terseness

gpt-4o-mini answers are noticeably **shorter** than Qwen3.5 9B's, frequently omitting secondary facts (specific prices, VAT lines, contact info, secondary conditions) even when the context contains them. The 12 partial results all share this pattern. Q6 is the only outright wrong answer — the model hallucinated stock instead of using the retrieved chunk.

Fix path: tighten `SYSTEM_PROMPT` in [config.py](rag-chatbot/config.py) — specifically rule 7 ("include all items found in the context with their key details") and rule 2 ("Include specific numbers, prices, EUR amounts..."). Adding a few-shot example showing the expected level of detail would likely close the gap.

---

## Eval 6: Best Local Run

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

### ✅ Resolved in Eval 9
- **Timeouts (Q6, Q8, Q15, Q20)** — gone entirely with gpt-4o-mini. Local Qwen3.5 9B on a 3070 couldn't generate long list-heavy answers within the 120s timeout; the OpenAI API has no such ceiling.
- **Q12 retrieval failure** — fixed by `text-embedding-3-small`. The 1536-dim multilingual model embeds "pre-purchase inspection" closely enough to the pricing chunk. Q12 retrieval went from 0% to 100%.
- **Q19 wrong answer (Overtime → Salary)** — masked by the FAQ entry added for overtime. Underlying retrieval has not been re-tested without the FAQ, but the symptom is resolved end-to-end.

### ✅ Resolved in Eval 10
- **Q6 SUV stock hallucination** — fixed by atomic chunking of `Vehicle-Stock-And-Inventory.txt`. The full 6391-char inventory is now retrieved as a single chunk, so the model sees all 8 SUVs at once. Q6 now lists every model with prices.
- **Closing pleasantries** ("feel free to reach out!", "I hope this helps!") — eliminated by an explicit ban in the rewritten `SYSTEM_PROMPT`.

### 1. Prompt-Driven Omission (Eval 11 confirms — dominant failure mode)
9 of 11 partials have facts visible in the retrieved chunks that the model omits. The clearest case is Q14: chunk 1 has front/rear/disc/fluid prices, chunk 5 has 21% VAT and 6-month warranty — yet the answer is one sentence about front pads only. The current `SYSTEM_PROMPT` enforces "list everything when asked to list" but allows the model to interpret single-fact questions narrowly. **Fix: prompt amendment + few-shot example demonstrating multi-fact answers to single-fact questions.** See "Next Steps P0".

### 2. Retrieval Gaps (Eval 11 confirms — secondary failure mode)
7 of 11 partials have at least one expected fact that is NOT in the retrieved chunks:
- Q9 EUR 20k delivery waiver
- Q10 Compensa Vienna, Gjensidige ADAS discount
- Q11 20-30 min appraisal, brand value commentary
- Q12 21% VAT line
- Q13 customs duty 6.5%, document list
- Q15 partial/full refund options, ODR/court escalation
- Q20 EUR 50/mo SEB pension match

The relevant chunks exist in source documents but rank 6-10 at the current TOP_K=5. **Fix: bump TOP_K to 8.** See "Next Steps P1".

### 3. Q20 Cross-Collection Leak
Retrieval pulls `Customer-Loyalty-And-Referral-Program.txt` (Gold tier service discounts) into the employee benefits answer. Both docs contain the words "Gold," "discount," "tier," and the loyalty doc lives in the employee collection because "employees see everything." **Fix: metadata-based audience filter.** See "Next Steps P2".

### 4. Q15 Topical Relevance vs Cosine Distance
The Test-Drive chunk "After the drive: the vehicle is inspected upon return…" appeared at distance 0.8074 in Q15 retrieval and polluted the answer with test-drive content. Cosine similarity rewarded vocabulary overlap (inspect, vehicle, return) over topical relevance. **Fix: cross-encoder re-ranker** (see "Next Steps P5"), or a stronger prompt instruction to ignore chunks that don't directly address the question.

### 5. File-Level Retrieval Precision Is No Longer Informative
The metric reports 100% across all recent runs because the right *file* is always among the retrieved sources — but the right *chunk within the file* is often missing (see §2). **Fix: replace with LLM-as-judge completeness scoring.** See "Next Steps P3".

---

## Next Steps (Prioritized by Eval 11 Diagnostic)

The Eval 11 chunk-level diagnostic gave us a clear picture: prompt engineering is the single highest-leverage fix, followed by a modest retrieval bump. The rest are second-order.

### P0 — Prompt engineering (fixes 9 of 11 partials)

The current `SYSTEM_PROMPT` enforces enumeration when the user asks for a list, but the model still drops secondary facts on single-fact questions (Q14 = pure terseness despite all facts being in chunks).

Concrete prompt amendment to add to [config.py](rag-chatbot/config.py):

> "After answering the direct question, scan the retrieved context for any other numeric facts on the same topic (alternative tier prices, related fees, VAT lines, secondary conditions, waivers, warranty windows, eligibility rules) and include them on a second line. Do not invent — only use facts explicitly in the context."

And add **one few-shot example** in the prompt showing a single-fact question (e.g. "How much is X?") producing a 3-4 sentence answer that includes the primary number plus related secondary facts from the context. Few-shot is more reliable than rule-based instructions for gpt-4o-mini.

**Expected wins after this fix:** Q8 (EV prices), Q14 (rear/disc/fluid/VAT/warranty), Q17 (family add-on), Q18 (sick leave), Q12 (OBD/Advanced), Q10 (partner names), Q9 (Vilnius free zone), Q15 (less Test-Drive leak).

### P1 — Bump TOP_K from 5 to 8 (fixes 5 retrieval gaps)

Of the 7 retrieval-component partials, 5 are likely solvable by retrieving 3 more chunks: Q9 EUR 20k waiver, Q11 appraisal/brand, Q13 customs/docs, Q15 refunds/escalation, Q20 SEB pension.

Risk: more chunks = more low-relevance content the model may use (Q15 already leaked from a 5th chunk). Mitigate by adding a distance threshold (drop chunks with `distance > 1.3`) — most usable chunks have distance < 1.0 in current evals; the dropped ones are noise anyway.

This is a one-line change in [config.py](rag-chatbot/config.py) (`TOP_K=8`) plus optional filtering in [query.py:retrieve_chunks](rag-chatbot/query.py#L119).

### P2 — Per-mode source filtering (fixes Q20 cross-collection leak)

The `Customer-Loyalty-And-Referral-Program.txt` doc shouldn't be retrieved when answering employee-mode questions about employee benefits. Right now the employee collection includes it because employees "see everything."

Two options:
- **Metadata filter** at query time: `where={"audience": {"$ne": "customer-loyalty"}}` for employee benefit queries. Requires tagging at ingest.
- **Split collections more strictly**: have the employee collection exclude customer-loyalty docs specifically, since "employees see everything" was the original intent but loyalty isn't an internal HR doc.

The metadata approach is cleaner — add an `audience` field to chunk metadata in [ingest.py](rag-chatbot/ingest.py) and let queries filter as needed.

### P3 — LLM-as-judge metric

The current "retrieval precision" metric is at 100% across the board and no longer differentiates runs. We need a metric that captures answer completeness:

- Send `(question, expected_answer, actual_answer)` triples to a stronger model (Claude Opus 4.7 or gpt-4o) and ask for a 0-5 completeness score and a list of facts in expected-but-not-in-actual.
- Cost is trivial (~$0.10 per eval run with Claude or gpt-4o).
- Replaces my manual "Correct/Partial/Wrong" tagging — which currently requires me to read every answer.

### P4 — Hyperparameter sweep (only if P0+P1 don't close the gap)

Only after the prompt fix and TOP_K bump, if results are still partial, run a small grid:
- `CHUNK_SIZE ∈ {400, 600, 800}` × `CHUNK_OVERLAP ∈ {60, 120}` × `TOP_K ∈ {5, 8, 12}`
- 9 configs × 20 questions × ~$0.02 per eval ≈ $4 total
- Don't run this blind — diagnose first, tune second.

### P5 — Long-term enhancements (lower priority)

- **Hybrid retrieval (BM25 + vector)**: would help if multilingual queries miss on semantic embeddings. Current Lithuanian + English mix performs fine at retrieval, so this is speculative for now.
- **Cross-encoder re-ranking**: `cross-encoder/ms-marco-MiniLM-L-6-v2` on top-k results. Could fix Q15 Test-Drive pollution (low cosine distance but low topical relevance).
- **Inventory FAQ entries**: pre-canned answers for the most common stock queries (SUVs, EVs, sedans by price band) — sidesteps both retrieval and prompt issues for these.
- **Dockerize**: `docker-compose.yml` for one-command setup.

## ✅ Completed

- **Chunk-level eval diagnostic** (Eval 11) — `retrieve_chunks()` returns ranked chunks with distance; eval reports include full chunk content. Made the prompt-vs-retrieval distinction observable.
- **Atomic chunking for inventory** (Eval 10) — `ATOMIC_FILES` in [config.py](rag-chatbot/config.py) + branch in [ingest.py:chunk_documents()](rag-chatbot/ingest.py). Q6 went from wrong to correct.
- **Switch to OpenAI embeddings** (Eval 9) — retrieval precision 92.5% → 100%.
- **Switch to OpenAI LLM** (Eval 9) — timeouts 4 → 0, answered 16/20 → 20/20.