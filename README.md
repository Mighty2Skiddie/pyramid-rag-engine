# Pranav Sharma — Vexoo Labs AI Engineer Assignment

A three-part system covering document ingestion with hierarchical retrieval, LLM fine-tuning for math reasoning, and a plug-and-play query routing architecture.

---

## What This Is

The core problem this assignment addresses is a real one: **how do you make a large document queryable without an LLM API call on every request?** The answer this implementation takes is to process the document once upfront — breaking it into chunks, enriching each chunk with summaries, categories, and keywords — and then retrieval becomes a fast similarity lookup rather than an expensive generation step.

The three components aren't arbitrary:
- **Part 1** (Document Pipeline + RAG) demonstrates retrieval system design
- **Part 2** (GSM8K Fine-Tuning) demonstrates adaptation of a foundation model for a specific task
- **Bonus** (Reasoning Adapter) demonstrates clean architectural thinking when behavior needs to vary by input type

---

## Quick Start

### Install

```bash
# Clone the repo, then install as an editable package
pip install -e .
```

This registers `part1_document_pipeline`, `bonus_reasoning_adapter`, and `shared` as proper Python packages — no `sys.path` manipulation needed anywhere.

### Run the Document Pipeline Demo

```bash
python examples/demo_pipeline.py
```

Loads the sample Nexagen Technologies annual report (~5KB), chunks it, builds the Knowledge Pyramid, runs 5 queries, then drops into interactive mode.

### Run Tests

```bash
python -m pytest tests/ -v
```

39 tests, all passing. Covers every layer: text utils, input loading, chunking, similarity functions, pyramid building, retrieval, and the reasoning adapter.

### Run the Reasoning Adapter Demo

```bash
python -m bonus_reasoning_adapter.adapter
```

### Part 2 (Fine-Tuning)

Designed for Google Colab with a T4 GPU. See [`part2_gsm8k_finetuning/README.md`](part2_gsm8k_finetuning/README.md) for setup instructions.

---

## Project Structure

```
├── part1_document_pipeline/
│   ├── input_layer.py       # Document loading — txt, pdf, raw string
│   ├── chunker.py           # Sliding window chunker with overlap
│   ├── pyramid_builder.py   # 4-layer Knowledge Pyramid construction
│   ├── retriever.py         # Multi-level semantic retrieval + scorer
│   ├── similarity.py        # TF-IDF, fuzzy, Jaccard, cosine utilities
│   └── models.py            # Dataclass-based data models
│
├── part2_gsm8k_finetuning/
│   ├── gsm8k_lora_finetuning.py   # Full training script (Colab)
│   └── README.md                  # Setup and expected outputs
│
├── bonus_reasoning_adapter/
│   ├── interfaces.py    # Abstract base classes (Strategy Pattern)
│   ├── classifier.py    # Keyword-based query type classifier
│   ├── handlers.py      # Domain-specific reasoning handlers
│   └── adapter.py       # Router + chain support + routing telemetry
│
├── shared/
│   ├── logger.py          # Structured logging, console + file handlers
│   ├── text_utils.py      # Normalization, sentence splitting, tokenization
│   └── config_manager.py  # Typed config with env-var overrides
│
├── tests/
│   └── test_pipeline.py   # Full test suite (39 tests)
│
├── examples/
│   ├── sample_document.txt   # Nexagen Technologies annual report (mock)
│   └── demo_pipeline.py      # End-to-end demo script
│
├── docs/
│   ├── system_design.md         # Architecture and data flow
│   ├── engineering_decisions.md # Why each design choice was made
│   └── summary_report.md        # One-page overview
│
├── pyproject.toml   # Package metadata and install config
└── requirements.txt # Pinned dependencies
```

---

## Part 1: Document Ingestion + Knowledge Pyramid

### The Core Idea

A naive RAG system chunks text, embeds it with a neural encoder, and does nearest-neighbor retrieval. That works, but it has a failure mode: the semantic embedding of a query about "quarterly revenue" might not resemble the embedding of a dense financial paragraph where "revenue" appears once buried in five other topics.

The Knowledge Pyramid approach solves this by enriching each chunk with multiple compressed representations. Retrieval then happens at all levels simultaneously, so the system can match a factual query at the keyword level (L4) and a conceptual query at the summary level (L2) — and combine both signals.

### The Four Layers

| Layer | What's stored | How it's built | Why |
|-------|--------------|----------------|-----|
| L1 — Raw Text | Original chunk | Stored as-is | Ground truth; used for TF-IDF lexical matching |
| L2 — Summary | 2-sentence extractive summary | Sentence frequency scoring (simplified TextRank) | Summaries match conceptual queries better than raw text |
| L3 — Category | Domain label (finance, legal, technical, medical) | Keyword dict matching | Fast coarse filter; prevents a finance query from ranking medical chunks highly |
| L4 — Distilled | Top TF-IDF keywords + 128-d embedding vector | scikit-learn TF-IDF + hash-seeded mock embedding | Most signal-dense representation; weighted highest in retrieval |

### Retrieval Scoring

Each query is scored against all four layers independently:

```
L1 score: TF-IDF cosine similarity (query vs. raw chunk text)
L2 score: Fuzzy token match (rapidfuzz token_set_ratio, handles word order)
L3 score: Category label match (1.0 if matching non-general category, else 0.0/0.3)
L4 score: 0.6 × Jaccard(query_keywords, chunk_keywords) + 0.4 × cosine(query_vec, chunk_vec)

Final score: 0.15·L1 + 0.30·L2 + 0.10·L3 + 0.45·L4
```

L4 is weighted highest because it's the most distilled signal — TF-IDF keywords capture the distinctive vocabulary of a chunk, and they're exact matches against what the user is likely searching for. L2 is next because summaries abstract away noise. L3 is low because category matching is too coarse to be the primary signal, but it's a useful tie-breaker.

### Configuration

Tunable at runtime via environment variables:

```bash
CHUNKER_WINDOW_SIZE=2500 RETRIEVER_TOP_K=5 python examples/demo_pipeline.py
```

---

## Part 2: GSM8K Fine-Tuning

### Setup

- **Model**: LLaMA 3.2 1B (auto-falls back to TinyLlama 1.1B if access token unavailable)
- **Method**: LoRA (r=8, α=16) targeting `q_proj` and `v_proj` attention projections
- **Quantization**: 4-bit NF4 (QLoRA) to fit on a T4 GPU (16GB VRAM)
- **Data**: 3,000 train / 1,000 test from GSM8K, shuffled with seed=42
- **Training**: 3 epochs, effective batch size 32 via gradient accumulation (4×8), fp16, cosine LR, AdamW

### What to Expect

```
Training time: ~20–30 minutes on T4 GPU
Exact match accuracy: ~25–35% (baseline zero-shot: ~5%)
Adapter size: ~10–50 MB (vs ~2 GB for full model weights)
```

---

## Bonus: Reasoning Adapter

A **Strategy Pattern** implementation for routing queries to domain-specific handlers at runtime.

```
User Query
    │
    ▼
KeywordQueryClassifier  ──→  (QueryType, confidence)
    │
    ▼
Handler Registry lookup
    │
    ├── MATH    → MathReasoningHandler     (chain-of-thought arithmetic)
    ├── LEGAL   → LegalReasoningHandler    (clause-aware, adds disclaimer)
    ├── FACTUAL → FactualReasoningHandler  (precision-first retrieval)
    ├── CODE    → CodeReasoningHandler     (technical context parsing)
    └── GENERAL → DefaultReasoningHandler  (standard RAG)
    │
    ▼
ReasoningResponse (answer + trace + confidence + metadata)
```

**Adding a new domain** requires only two steps:
1. Create a class implementing `ReasoningHandler`
2. Call `adapter.register_handler(MyHandler())`

No changes to the routing logic, the classifier, or any existing handler. The classifier is itself swappable — the `QueryClassifier` abstract class means you can drop in a fine-tuned DistilBERT classifier without touching the adapter or handlers.

Handlers also support **chaining** — passing one handler's answer as context to the next. The `route_with_chain(query, [FACTUAL, MATH])` pattern is designed for queries like "calculate X% of the company's revenue" where you need retrieval before arithmetic.

---

## Design Decisions

See [`docs/engineering_decisions.md`](docs/engineering_decisions.md) for the full reasoning. Key choices:

1. **Rule-based pyramid layers over LLM calls** — Zero external dependencies, fully reproducible, free to run. The architecture is modular enough that any layer can be upgraded to an LLM-based version independently.

2. **Multi-level weighted retrieval** — Different query types match best at different abstraction levels. A single embedding score misses this. The weighted sum captures complementary signals.

3. **LoRA over full fine-tuning** — Reduces trainable parameters from 1B to ~20M, produces a portable 10–50MB adapter, and prevents catastrophic forgetting on tasks outside GSM8K.

4. **Strategy Pattern over if/else routing** — The classifier output is data, not control flow. This makes adding domains trivial and the routing logic independently testable.

---

## Limitations

- **Mock embeddings**: The 128-dimensional vectors are hash-seeded pseudo-random. They produce consistent results but no real semantic similarity. For production, swap `generate_mock_embedding()` with `sentence-transformers`.
- **In-memory pyramid**: The index lives entirely in RAM. For large document collections, this needs to move to a vector store (Chroma, Weaviate, Pinecone) for persistence and scalability.
- **Rule-based classifier**: The keyword classifier in the reasoning adapter will mis-route ambiguous queries. It's a demonstration of the routing architecture, not a production classifier.
- **Part 2 expected accuracy**: 25–35% exact match on GSM8K is typical for a 1B model with 3 epochs. Larger models (7B+) or longer training can push this to 60%+, but that's outside free-tier Colab constraints.

---

## Technologies

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Similarity | scikit-learn TF-IDF | Lightweight, no GPU, well-tested |
| Fuzzy match | rapidfuzz | 10–100× faster than fuzzywuzzy, MIT licensed |
| PDF parsing | PyMuPDF | More reliable text extraction than pdfminer for complex layouts |
| Fine-tuning | HuggingFace peft + trl | Standard LoRA ecosystem, native HF integration |
| Config | Python dataclasses + env vars | No YAML/TOML parsing overhead; env vars enable runtime tuning |
| Testing | pytest + unittest | unittest for class-based test organization; pytest for runner |
