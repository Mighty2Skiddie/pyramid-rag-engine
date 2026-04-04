# System Design — Vexoo Labs AI Assignment

## The Architecture at a Glance

This system has three logical layers that correspond to the three assignment parts:

1. **Document Intelligence Layer** (Part 1) — takes a raw document and makes it queryable
2. **Math Reasoning Layer** (Part 2) — a fine-tuned LLM for arithmetic chain-of-thought
3. **Query Routing Layer** (Bonus) — routes queries to the right handler based on type

They share a common utility layer for logging, config, and text processing.

---

## Layer Architecture

### Input Layer
**Responsibility**: Accept raw input in any supported format and normalize it into a consistent internal representation.

`input_layer.py` handles three input types — plain strings, `.txt`/`.md` files, and `.pdf` files — and produces a `Document` dataclass with a deterministic content-hash ID. The key design choice here is that format handling is **contained entirely in this layer**. Nothing downstream knows or cares whether the text came from a PDF or was passed as a string. Adding DOCX support means touching one function (`_load_from_file`) and nothing else.

PDF extraction uses PyMuPDF rather than pdfminer because it handles multi-column layouts and embedded fonts more reliably. The trade-off is a heavier dependency, but PDF quality in real documents (especially annual reports) varies enormously and PyMuPDF fails less often.

### Processing Layer
**Responsibility**: Transform a flat document into a structured, multi-representation knowledge index.

This is where most of the complexity lives. Two components:

**Chunker** (`chunker.py`): Straightforward sliding window over characters. Window size defaults to 2,000 characters (~2 pages of dense text); overlap is 15% (300 characters). The overlap ensures that content near chunk boundaries isn't systematically missed — a sentence split across a boundary still appears complete in one of the two adjacent chunks.

Character-based (rather than token-based) chunking was chosen to avoid a tokenizer dependency. The estimate `token_count ≈ chars / 4` is rough but sufficient for the metadata it populates.

**Pyramid Builder** (`pyramid_builder.py`): Processes each chunk into four representations:
- L1 is the raw text stored as-is
- L2 is an extractive summary using word-frequency sentence ranking (simplified TextRank)
- L3 is a category label assigned by keyword matching against domain dictionaries
- L4 is a ranked keyword list from TF-IDF (fit once on the full corpus for consistent vocabulary) plus a hash-seeded embedding vector

The TF-IDF vectorizer is fit on the full set of chunks before individual nodes are built. This matters: if you fit a separate vectorizer per chunk, you lose the ability to compare keyword scores across chunks because each has its own vocabulary. Fitting once on the corpus gives a shared feature space.

### Data Layer
**Responsibility**: Hold the in-memory index and provide retrieval access.

The pyramid index is a Python dict: `chunk_id → PyramidNode`. There's no database, no vector store, no persistence. This is intentional for assignment scope — the system is designed so the pyramid serializes cleanly to JSON or pickle, but that step wasn't implemented. The index is rebuilt on each run.

For production, this is the first thing to change. A persistent vector store (Chroma for local, Pinecone for cloud) would hold the L4 embeddings. The L1–L3 representations could live in SQLite or a document DB.

### Service Layer
**Responsibility**: Execute queries against the index and return ranked results.

`retriever.py` computes four independent similarity scores per chunk (TF-IDF cosine for L1, fuzzy token match for L2, category label match for L3, keyword Jaccard + vector cosine for L4), combines them with fixed weights, filters by a minimum confidence threshold, and returns the top-K results with full provenance metadata.

The scoring is intentionally transparent — every result includes a `level_scores` dict showing exactly how much each layer contributed. This is useful for debugging: if a "clearly relevant" chunk is ranking low, you can see whether it's because the keywords aren't matching (L4 issue) or the category is wrong (L3 issue).

The reasoning adapter (`adapter.py`) sits at the service layer boundary — it decides which reasoning strategy to apply before or alongside retrieval.

### Output Layer
**Responsibility**: Format results for the caller.

`format_results()` in `retriever.py` and `ReasoningResponse` in the adapter both produce structured output. The choice to return dataclasses (rather than dict or raw strings) was deliberate: callers get type safety and IDE autocompletion, and the structure is explicit in the code rather than implicit in documentation.

---

## Data Flow — Part 1

```
 Raw Document (txt / pdf / string)
          │
          ▼
 InputLayer.load_document()
   → Normalize encoding (UTF-8, strip control chars)
   → Deterministic doc_id from MD5(content)
   → Document(doc_id, text, source, char_count, token_count)
          │
          ▼
 chunk_document(document, config)
   → Sliding window: step = window_size × (1 - overlap_ratio)
   → Content deduplication via MD5 per chunk
   → List[Chunk(chunk_id, doc_id, text, start_char, end_char, token_count)]
          │
          ▼
 build_pyramid(chunks, config)
   → Fit TF-IDF vectorizer once on all chunk texts
   → Per chunk:
       L1: raw_text = chunk.text
       L2: summary = top-2 frequency-scored sentences
       L3: (category, confidence) = keyword match against domain dicts
       L4: keywords = top-10 TF-IDF terms
           embedding = SHA-256 seeded RNG → unit-normalized 128-d vector
   → Dict[chunk_id, PyramidNode]
          │
          ▼ (index ready)

 User Query
          │
          ▼
 query_pyramid(query_text, pyramid_index, config)
   → Compute L1 scores: TF-IDF cosine(query, all raw_texts)
   → Compute L2 scores: fuzzy token_set_ratio(query, each summary)
   → Compute L3 scores: category label match
   → Compute L4 scores: Jaccard(query_kws, chunk_kws) + cosine(query_emb, chunk_emb)
   → Weighted sum: 0.15·L1 + 0.30·L2 + 0.10·L3 + 0.45·L4
   → Filter by min_confidence_threshold
   → Sort descending, return top_k
          │
          ▼
 List[QueryResult(chunk_id, score, best_level, level_scores, raw_text, summary, ...)]
```

---

## Data Flow — Part 2

```
 GSM8K (HuggingFace)
   → Shuffle(seed=42) → 3,000 train / 1,000 test
          │
          ▼
 format_gsm8k_prompt(example)
   → "Question: {q}\nAnswer: Let's solve this step by step.\n{chain_of_thought}#### {answer}"
          │
          ▼
 Tokenizer (LLaMA 3.2 1B)
   → max_length=512, truncation, padding
   → labels = input_ids (causal LM objective)
          │
          ▼
 4-bit NF4 quantized base model
 + LoRA adapters (r=8, α=16, q_proj + v_proj)
          │
          ▼
 HuggingFace Trainer
   → 3 epochs, batch=4, grad_accum=8 (effective batch=32)
   → AdamW, cosine LR schedule, fp16
   → Checkpoints at each epoch, load best on eval_loss
          │
          ▼
 Evaluation on 100 test samples
   → Greedy decoding (temperature=0.0)
   → Regex: r'####\s*(\-?\d[\d,]*\.?\d*)' to extract predicted answer
   → Exact match against ground truth
          │
          ▼
 Artifacts: adapter weights (~10-50MB) + metrics JSON
```

---

## State Management

**Part 1**: All state is held in the `pyramid_index` dict for the lifetime of a process. There is no persistent state between runs. `SessionState` (in `models.py`) tracks what's been loaded in the current session — it's used by the demo script to show progress, not by retrieval logic.

**Part 2**: State lives in the Trainer and checkpoint files. The best checkpoint by eval_loss is loaded at the end of training. The LoRA adapter weights are the only artifacts that persist.

**Bonus Adapter**: The `routing_log` list in `ReasoningAdapter` accumulates per-query routing decisions for the lifetime of the adapter instance. This is an in-memory telemetry store — `get_routing_stats()` aggregates it. In production, this would write to a structured log or metrics store.

---

## Performance Considerations

**Pyramid build time** is O(n × m) where n is the number of chunks and m is the vocabulary size (capped at 5,000 for TF-IDF). For a 50KB document producing ~25 chunks, this takes under a second. At 500 chunks from a long document, it's still fast — TF-IDF is vectorized via scikit-learn's sparse matrix operations.

**Query time** is O(n) — one similarity computation per chunk. This is proportional to index size and will become a bottleneck somewhere around 10,000+ chunks. The fix is approximate nearest-neighbor search (FAISS, HNSW) for the L4 vector component, and inverted indexing for L1/L4 keyword components.

**Fuzzy matching** (L2) is the most expensive per-query operation — `rapidfuzz.fuzz.token_set_ratio` is fast but still O(n) with non-trivial string operations per chunk. For large indexes, L2 matching should be pre-filtered by L3 category to reduce the candidate set.

---

## Scalability Analysis

| Scale | Bottleneck | Fix |
|-------|-----------|-----|
| < 1,000 chunks | None — all in memory, fast | Current approach works |
| 1,000–10,000 chunks | Query latency (linear scan) | FAISS index for L4 vectors; inverted index for L1 |
| 10,000+ chunks | Memory (pyramid in RAM) | Move index to Chroma/Weaviate; persistent L1–L3 in SQLite |
| Multi-user / API | Concurrent reads (not thread-safe) | Move to FastAPI with async handlers; stateless query endpoints |
| Multiple documents | No cross-document retrieval boundary | Add doc_id filter to retrieval config |

---

## Technology Choices

| Component | Chosen | Considered | Why Chosen |
|-----------|--------|------------|-----------|
| Similarity | scikit-learn TF-IDF | gensim, BM25 | Well-documented, sparse matrix ops, no extra deps |
| Fuzzy match | rapidfuzz | fuzzywuzzy | 10–100× faster; fuzzywuzzy is literally a wrapper around rapidfuzz now |
| Vector ops | NumPy | FAISS, scipy | Sufficient for index size; FAISS is the upgrade path |
| Embeddings | Hash-seeded mock | sentence-transformers | Zero-dependency for assignment; clear swap point marked in code |
| PDF | PyMuPDF (fitz) | pdfminer, pypdf | More robust on complex layouts; handles embedded fonts |
| LLM training | HuggingFace Trainer | PyTorch Lightning | Native HF ecosystem; less boilerplate for standard SFT |
| LoRA | peft library | Manual LoRA | Standard implementation; peer-reviewed and tested at scale |
| Config | dataclasses + env vars | Pydantic, Hydra | No dependencies at this scale; upgrade path is clear |
