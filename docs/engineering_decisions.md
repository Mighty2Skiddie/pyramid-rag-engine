# Engineering Decisions — Vexoo Labs AI Assignment

This document explains the *why* behind the major technical choices in this project. The goal isn't to justify every line — it's to be honest about what trade-offs were made, what alternatives were considered, and what I'd change with more time or a larger scope.

---

## 1. Why TF-IDF Over Real Embeddings (Part 1)

The obvious choice for a retrieval system in 2024 is a dense embedding model — `sentence-transformers/all-MiniLM-L6-v2` is fast, produces genuinely semantic vectors, and runs on CPU in reasonable time.

I chose TF-IDF for the pyramid's L4 layer for two reasons:

**Reproducibility**: TF-IDF is deterministic. Given the same corpus and query, you always get the same weights. With `sentence-transformers`, there are CUDA non-determinism issues, model version drift, and you need the model to be downloaded before anything runs. For a self-contained assignment submission, determinism matters.

**Transparency**: TF-IDF scores are human-interpretable — you can look at the feature weights and immediately understand why a chunk scored the way it did. Neural embeddings are opaque. For debugging retrieval behavior, this matters more than marginal accuracy gains.

**The honest trade-off**: Real embeddings would significantly improve semantic recall — especially for paraphrased queries where the exact words don't appear in the chunk. The `generate_mock_embedding()` function exists precisely to mark this swap point clearly. In production, you'd replace that function's body with `model.encode(text)` and everything downstream would still work.

---

## 2. Why Multi-Level Retrieval Instead of a Single Score

The standard RAG approach: embed the query, find nearest chunks by cosine distance. Simple, fast, and often good enough.

The problem is that it's a single signal. A query like "what is the company's revenue growth?" might not be semantically close to a dense paragraph that buries "23% year-over-year" alongside a dozen other figures — especially with hash-seeded mock embeddings.

By computing similarity at four abstraction levels and combining them:
- L1 (raw text, TF-IDF) handles exact lexical matches
- L2 (summaries, fuzzy) handles paraphrased or conceptual queries
- L3 (category labels) provides a fast domain filter
- L4 (keywords + embedding) provides the densest signal

The weighted combination is deliberately tuned: L4 at 45% because keywords are the highest-signal distillation of a chunk's topic, L2 at 30% because summaries abstract away noise, L1 at 15% because raw text is noisy but captures exact matches, L3 at 10% because category matching is coarse but useful for domain-specific queries.

Could these weights be learned? Yes — you'd treat it as a learning-to-rank problem with labeled query-chunk pairs. The static weights are a reasonable starting point that can be empirically tuned.

---

## 3. Why Extractive Summaries for L2 (Not LLM-Based)

An LLM-based abstractive summary (e.g., FLAN-T5, GPT-4) would produce better, more coherent summaries. The extractive approach — scoring sentences by word frequency and picking the top two — is a simplified TextRank.

The trade-off is clear: extractive summaries are cheaper, faster, and don't require an API key. They also preserve the exact wording of the source text, which means TF-IDF and fuzzy matching against them still works well. Abstractive summaries might rephrase "capital expenditure" as "investment spending," which could hurt lexical similarity scores.

The `_generate_summary()` function in `pyramid_builder.py` is deliberately isolated with a clear docstring directing where to swap in an LLM call. This kind of "upgrade path comment" is more useful than making the system complex upfront.

---

## 4. Why LoRA r=8 Specifically (Part 2)

LoRA (Low-Rank Adaptation) inserts trainable rank-r matrices into the target layers. The rank r controls the capacity of the adapter — higher r means more parameters and more expressive power, but also more compute and more risk of overfitting.

For a 1B parameter model on 3,000 samples:
- `r=4` is probably underfitting — not enough capacity to learn the GSM8K reasoning pattern
- `r=16` or `r=32` would work but adds ~4× the parameters with diminishing returns on a small dataset
- `r=8` hits the commonly-cited sweet spot for this model size: ~16M trainable parameters out of 1B total (~1.6%)

The `alpha=16` (2× rank) is a standard heuristic from the original LoRA paper — it controls the scaling of the adapter output and effectively sets the learning rate for the adapter relative to the base model.

Targeting only `q_proj` and `v_proj` (not `k_proj`, `o_proj`, or the MLP layers) is a deliberate VRAM constraint. The T4 GPU has 16GB. Adding more target modules would push past the memory limit when combined with 4-bit quantization and fp16 gradients.

---

## 5. Why QLoRA (4-bit) Over fp16 Base Model

Full fp16 fine-tuning of a 1B parameter model requires roughly 12–16GB just for the model weights, plus optimizer states and gradients. A T4 has 16GB total.

QLoRA quantizes the base model to 4-bit NF4, reducing the base model footprint to ~500MB. The LoRA adapters train in bf16. This is the technique from Dettmers et al. (2023) and is now standard practice for fine-tuning on consumer hardware.

The accuracy cost is minimal — the base model is frozen and only the small adapter is trained. The 4-bit quantization affects inference precision, but the adapter learns to compensate.

---

## 6. Why the Strategy Pattern for the Reasoning Adapter

The most direct implementation would be an if/else chain:

```python
def route(query):
    if "calculate" in query or "sum" in query:
        return math_handler(query)
    elif "law" in query or "legal" in query:
        return legal_handler(query)
    ...
```

This works at first, but the moment you add a fifth domain, you're editing the routing function. And the moment a colleague wants to add domain detection through ML instead of keywords, they're touching the same function.

The Strategy Pattern separates three concerns that change for different reasons:
1. **How to classify** (keyword rules today, ML model tomorrow) — lives in `QueryClassifier`
2. **How to handle a domain** (templates today, LLM calls tomorrow) — lives in each `ReasoningHandler`
3. **How to route** (dispatch from type to handler) — lives in `ReasoningAdapter`

Each of those three things can change independently without touching the others. That's the Open/Closed Principle in practice, not in theory.

The `route_with_chain()` method is a bonus feature that shows the architecture's extensibility — sequential handler chains enable multi-step reasoning workflows without introducing a state machine.

---

## 7. Why Dataclasses for Config (Not Pydantic or Hydra)

Pydantic would give runtime validation, type coercion, and .env file support. Hydra would give hierarchical config composition and CLI overrides out of the box. Both are excellent tools.

For this scale — three subsystems, ten config parameters — the overhead of either framework isn't justified. Python dataclasses with `field(default_factory=...)` give type safety at the IDE level and require zero additional dependencies.

The `load_config()` function in `shared/config_manager.py` reads environment variables as overrides. This is the simplest possible approach to runtime configurability that doesn't require a config file format. The upgrade path to YAML + Pydantic is straightforward when the number of parameters grows.

---

## 8. Edge Cases and Known Failure Modes

**The classifier will mis-route ambiguous queries.** "How much tax does a legal entity pay?" has both `finance` and `legal` signals. The classifier takes the one with more keyword matches — it'll get this wrong sometimes. Production would need a confidence threshold below which the query goes to a human or a more expensive ML classifier.

**The pyramid breaks down on very short chunks.** The `min_chunk_size=100` guard prevents it, but if a document has one-sentence sections, the L2 summary will just be that one sentence — which adds no information over L1.

**TF-IDF vocabulary is corpus-specific.** The vectorizer is fit on the document being indexed. If the test queries use vocabulary that doesn't appear anywhere in the document, TF-IDF will score zero for every chunk. Real semantic embeddings handle this via distributional similarity — words that appear in similar contexts get similar representations.

**Mock embeddings have no semantic meaning.** Two chunks about revenue and profit will have unrelated hash-seeded vectors. This means the vector cosine component of L4 scoring is essentially random noise. In production, the embedding score would be the primary signal rather than a secondary one.

---

## 9. What I'd Change With More Time

**Replace mock embeddings with sentence-transformers.** This is the highest-impact single change. The `generate_mock_embedding()` function has a clear swap point — the rest of the pipeline is already built to handle real 128-d (or 384-d) vectors.

**Persist the pyramid to disk.** Currently the pyramid is rebuilt from scratch on every run. A document that's been processed once should be serialized (pickle or JSON) and reloaded. This is especially important for the PDF use case where extraction is slow.

**Add a re-ranker pass.** The current scoring is a weighted average of independent signals. A cross-encoder re-ranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) applied to the top-10 candidates would significantly improve precision without needing to process every chunk.

**LoRA on all attention projections.** Adding `k_proj` and `o_proj` as targets and using a larger r (e.g., 16 or 32) with gradient checkpointing would likely push GSM8K accuracy from ~30% toward ~45% without exceeding T4 memory.
