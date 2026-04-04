# Summary Report — Vexoo Labs AI Engineer Assignment
**Pranav Sharma**

---

## Part 1: Document Ingestion + Knowledge Pyramid

**Approach**: Sliding window chunker (2000 chars, 15% overlap) → 4-layer pyramid per chunk:

| Layer | Content | Method |
|-------|---------|--------|
| L1 | Raw text | Stored as-is |
| L2 | Summary | Frequency-ranked sentence extraction |
| L3 | Category | Keyword dict matching (finance / legal / technical / medical) |
| L4 | Keywords + embedding | TF-IDF (fit on full corpus) + hash-seeded 128-d vector |

**Retrieval**: Query scored against all 4 layers simultaneously. Weighted aggregation: L4 45% · L2 30% · L1 15% · L3 10%. Top-K results returned with per-level score breakdown.

---

## Part 2: GSM8K Fine-Tuning

- **Model**: LLaMA 3.2 1B (fallback: TinyLlama 1.1B) · **Platform**: Google Colab T4 GPU
- **Method**: LoRA (r=8, α=16) on `q_proj` + `v_proj` · 4-bit NF4 quantization (QLoRA)
- **Data**: 3000 train / 1000 test · **Training**: 3 epochs, effective batch 32 (4×8 grad accum), fp16, cosine LR
- **Eval**: Regex extract `#### <num>` → exact match · **Expected**: ~25–35% accuracy

---

## Bonus: Reasoning Adapter

Strategy Pattern router — keyword classifier detects domain (MATH / LEGAL / FACTUAL / CODE / GENERAL) → dispatches to registered handler. New domains added by registration only, zero changes to existing code. Handlers support sequential chaining.

---

## Key Decisions & Assumptions

- **No LLM API calls in Part 1** — fully self-contained; rule-based methods keep it reproducible and free to run
- **Mock embeddings** — hash-seeded vectors; real semantic embeddings (sentence-transformers) are a direct swap at `generate_mock_embedding()`
- **QLoRA for memory** — 4-bit quantization fits the 1B model on 16GB VRAM; trades minor precision for trainability on free hardware
- **Multi-level retrieval** — single-score retrieval misses query types that match better at different abstraction levels
