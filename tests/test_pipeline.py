"""
Test suite for Part 1 — Document Ingestion Pipeline.

Tests cover:
    - Input layer: document loading and normalization
    - Chunker: sliding window behavior and edge cases
    - Pyramid builder: all 4 layers generated correctly
    - Retriever: query returns ranked results
    - Similarity: all similarity functions
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np

# Locate the project root relative to this file (tests/ → project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

from part1_document_pipeline.input_layer import load_document
from part1_document_pipeline.chunker import chunk_document
from part1_document_pipeline.pyramid_builder import build_pyramid
from part1_document_pipeline.retriever import query_pyramid, format_results
from part1_document_pipeline.similarity import (
    compute_tfidf_similarity,
    fuzzy_match_score,
    keyword_jaccard_score,
    vector_cosine_similarity,
    generate_mock_embedding
)
from part1_document_pipeline.models import Document, Chunk
from shared.config_manager import ChunkerConfig, PyramidConfig, RetrieverConfig
from shared.text_utils import normalize_text, extract_sentences, tokenize_simple


# ────────────────────────────────────────────────────────────────
#  Test: Text Utilities
# ────────────────────────────────────────────────────────────────

class TestTextUtils(unittest.TestCase):
    """Tests for shared text utility functions."""
    
    def test_normalize_text_whitespace(self):
        text = "Hello   world\t\t  foo"
        result = normalize_text(text)
        self.assertEqual(result, "Hello world foo")
    
    def test_normalize_text_empty(self):
        self.assertEqual(normalize_text(""), "")
        self.assertEqual(normalize_text(None), "")
    
    def test_extract_sentences(self):
        text = "First sentence. Second sentence. Third sentence."
        sentences = extract_sentences(text)
        self.assertEqual(len(sentences), 3)
    
    def test_extract_sentences_empty(self):
        self.assertEqual(extract_sentences(""), [])
    
    def test_tokenize_simple(self):
        tokens = tokenize_simple("The quick brown fox jumps")
        self.assertNotIn("the", tokens)  # Stop word removed
        self.assertIn("quick", tokens)


# ────────────────────────────────────────────────────────────────
#  Test: Input Layer
# ────────────────────────────────────────────────────────────────

class TestInputLayer(unittest.TestCase):
    """Tests for document loading and normalization."""
    
    def test_load_string_input(self):
        doc = load_document("This is a test document with some content.")
        self.assertIsNotNone(doc.doc_id)
        self.assertEqual(doc.source, "string_input")
        self.assertGreater(doc.char_count, 0)
    
    def test_load_string_deterministic_id(self):
        doc1 = load_document("Same content here")
        doc2 = load_document("Same content here")
        self.assertEqual(doc1.doc_id, doc2.doc_id)  # Same content → same ID
    
    def test_load_empty_raises(self):
        with self.assertRaises(ValueError):
            load_document("")
    
    def test_load_txt_file(self):
        sample_path = _PROJECT_ROOT / "examples" / "sample_document.txt"
        if sample_path.exists():
            doc = load_document(str(sample_path))
            self.assertGreater(doc.char_count, 100)
            self.assertIn("sample_document.txt", doc.source)


# ────────────────────────────────────────────────────────────────
#  Test: Sliding Window Chunker
# ────────────────────────────────────────────────────────────────

class TestChunker(unittest.TestCase):
    """Tests for the sliding window chunking logic."""
    
    def setUp(self):
        self.doc = Document(
            doc_id="test_doc",
            text="A" * 5000,  # 5000 chars
            source="test",
            char_count=5000,
            token_count=1250
        )
    
    def test_basic_chunking(self):
        config = ChunkerConfig(window_size_chars=2000, overlap_ratio=0.15)
        chunks = chunk_document(self.doc, config)
        self.assertGreater(len(chunks), 1)
    
    def test_overlap_produces_more_chunks(self):
        config_no_overlap = ChunkerConfig(window_size_chars=2000, overlap_ratio=0.0)
        config_overlap = ChunkerConfig(window_size_chars=2000, overlap_ratio=0.15)
        
        chunks_no = chunk_document(self.doc, config_no_overlap)
        chunks_yes = chunk_document(self.doc, config_overlap)
        
        self.assertGreaterEqual(len(chunks_yes), len(chunks_no))
    
    def test_single_window_document(self):
        small_doc = Document(
            doc_id="small", text="Short text", source="test",
            char_count=10, token_count=3
        )
        chunks = chunk_document(small_doc)
        self.assertEqual(len(chunks), 1)
    
    def test_empty_document(self):
        empty_doc = Document(
            doc_id="empty", text="", source="test",
            char_count=0, token_count=0
        )
        chunks = chunk_document(empty_doc)
        self.assertEqual(len(chunks), 0)
    
    def test_chunk_ids_unique(self):
        chunks = chunk_document(self.doc)
        ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))  # All unique
    
    def test_chunk_positions_valid(self):
        chunks = chunk_document(self.doc)
        for chunk in chunks:
            self.assertGreaterEqual(chunk.start_char, 0)
            self.assertLessEqual(chunk.end_char, self.doc.char_count)
            self.assertLess(chunk.start_char, chunk.end_char)


# ────────────────────────────────────────────────────────────────
#  Test: Similarity Functions
# ────────────────────────────────────────────────────────────────

class TestSimilarity(unittest.TestCase):
    """Tests for all similarity computation functions."""
    
    def test_tfidf_similarity_basic(self):
        query = "machine learning algorithm"
        docs = [
            "machine learning is a subset of artificial intelligence",
            "cooking recipes for healthy meals",
            "deep learning algorithms for NLP"
        ]
        scores = compute_tfidf_similarity(query, docs)
        self.assertEqual(len(scores), 3)
        # ML doc should score higher than cooking doc
        self.assertGreater(scores[0], scores[1])
    
    def test_fuzzy_match(self):
        score = fuzzy_match_score("financial report", "financial annual report 2024")
        self.assertGreater(score, 0.5)
    
    def test_fuzzy_match_empty(self):
        self.assertEqual(fuzzy_match_score("", "text"), 0.0)
    
    def test_keyword_jaccard(self):
        score = keyword_jaccard_score(
            ["revenue", "profit", "growth"],
            ["revenue", "growth", "market", "earnings"]
        )
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_keyword_jaccard_no_overlap(self):
        score = keyword_jaccard_score(["a", "b"], ["c", "d"])
        self.assertEqual(score, 0.0)
    
    def test_vector_cosine_identical(self):
        vec = np.array([1.0, 2.0, 3.0])
        score = vector_cosine_similarity(vec, vec)
        self.assertAlmostEqual(score, 1.0, places=5)
    
    def test_mock_embedding_deterministic(self):
        emb1 = generate_mock_embedding("hello world")
        emb2 = generate_mock_embedding("hello world")
        np.testing.assert_array_equal(emb1, emb2)
    
    def test_mock_embedding_different_texts(self):
        emb1 = generate_mock_embedding("hello world")
        emb2 = generate_mock_embedding("goodbye moon")
        self.assertFalse(np.array_equal(emb1, emb2))
    
    def test_mock_embedding_normalized(self):
        emb = generate_mock_embedding("some text")
        norm = np.linalg.norm(emb)
        self.assertAlmostEqual(norm, 1.0, places=5)


# ────────────────────────────────────────────────────────────────
#  Test: Knowledge Pyramid Builder
# ────────────────────────────────────────────────────────────────

class TestPyramidBuilder(unittest.TestCase):
    """Tests for the 4-layer Knowledge Pyramid."""
    
    def setUp(self):
        self.chunks = [
            Chunk(
                chunk_id="test_chunk_001", doc_id="test_doc",
                text=(
                    "The company reported total revenue of $847 million. "
                    "Operating profit margins improved to 18.4 percent. "
                    "Capital expenditure totaled $156 million for data centers."
                ),
                start_char=0, end_char=200, token_count=50
            ),
            Chunk(
                chunk_id="test_chunk_002", doc_id="test_doc",
                text=(
                    "The engineering team deployed 47 major product updates. "
                    "The core platform uses a microservices architecture. "
                    "API latency was reduced from 230ms to 45ms."
                ),
                start_char=200, end_char=400, token_count=50
            )
        ]
    
    def test_pyramid_builds_all_chunks(self):
        pyramid = build_pyramid(self.chunks)
        self.assertEqual(len(pyramid), 2)
        self.assertIn("test_chunk_001", pyramid)
        self.assertIn("test_chunk_002", pyramid)
    
    def test_pyramid_has_all_layers(self):
        pyramid = build_pyramid(self.chunks)
        node = pyramid["test_chunk_001"]
        
        # L1: Raw text
        self.assertGreater(len(node.raw_text), 0)
        # L2: Summary
        self.assertGreater(len(node.summary), 0)
        # L3: Category
        self.assertIn(node.category, ["finance", "legal", "technical", "medical", "general"])
        # L4: Keywords
        self.assertGreater(len(node.keywords), 0)
        # L4: Embedding
        self.assertIsNotNone(node.embedding)
        self.assertEqual(len(node.embedding), 128)
    
    def test_finance_chunk_classified_correctly(self):
        pyramid = build_pyramid(self.chunks)
        node = pyramid["test_chunk_001"]
        # The finance chunk should be classified as finance
        self.assertEqual(node.category, "finance")
    
    def test_tech_chunk_classified_correctly(self):
        pyramid = build_pyramid(self.chunks)
        node = pyramid["test_chunk_002"]
        self.assertEqual(node.category, "technical")
    
    def test_empty_chunks(self):
        pyramid = build_pyramid([])
        self.assertEqual(len(pyramid), 0)


# ────────────────────────────────────────────────────────────────
#  Test: Retriever
# ────────────────────────────────────────────────────────────────

class TestRetriever(unittest.TestCase):
    """Tests for the query interface and semantic retriever."""
    
    def setUp(self):
        chunks = [
            Chunk("c1", "doc", "Revenue grew 23% with profits reaching new highs. Market cap exceeded $10 billion.", 0, 100, 25),
            Chunk("c2", "doc", "The software architecture uses microservices with API gateway. System deploys on cloud infrastructure.", 100, 200, 25),
            Chunk("c3", "doc", "Legal compliance with GDPR and CCPA regulations. Contract liability clauses were updated.", 200, 300, 25),
        ]
        self.pyramid = build_pyramid(chunks)
    
    def test_query_returns_results(self):
        results = query_pyramid("What is the revenue?", self.pyramid)
        self.assertGreater(len(results), 0)
    
    def test_query_results_sorted(self):
        results = query_pyramid("financial performance", self.pyramid)
        if len(results) > 1:
            self.assertGreaterEqual(results[0].score, results[1].score)
    
    def test_query_result_has_metadata(self):
        results = query_pyramid("software architecture", self.pyramid)
        if results:
            r = results[0]
            self.assertIsNotNone(r.chunk_id)
            self.assertIsNotNone(r.best_level)
            self.assertIn(r.best_level, ["L1", "L2", "L3", "L4"])
    
    def test_empty_query(self):
        results = query_pyramid("", self.pyramid)
        self.assertEqual(len(results), 0)
    
    def test_format_results(self):
        results = query_pyramid("revenue", self.pyramid)
        output = format_results(results, "revenue")
        self.assertIn("revenue", output.lower())


# ────────────────────────────────────────────────────────────────
#  Test: Reasoning Adapter (Bonus)
# ────────────────────────────────────────────────────────────────

class TestReasoningAdapter(unittest.TestCase):
    """Tests for the bonus reasoning adapter routing system."""
    
    def test_math_routing(self):
        from bonus_reasoning_adapter.adapter import ReasoningAdapter
        adapter = ReasoningAdapter()
        response = adapter.route("What is the sum of 10 and 20?")
        self.assertEqual(response.query_type.value, "math")
        self.assertEqual(response.handler_name, "MathReasoningHandler")
    
    def test_legal_routing(self):
        from bonus_reasoning_adapter.adapter import ReasoningAdapter
        adapter = ReasoningAdapter()
        response = adapter.route("What does the law say about liability?")
        self.assertEqual(response.query_type.value, "legal")
    
    def test_general_fallback(self):
        from bonus_reasoning_adapter.adapter import ReasoningAdapter
        adapter = ReasoningAdapter()
        response = adapter.route("Tell me something interesting")
        self.assertEqual(response.query_type.value, "general")
    
    def test_math_handler_computes(self):
        from bonus_reasoning_adapter.adapter import ReasoningAdapter
        adapter = ReasoningAdapter()
        response = adapter.route("Calculate the sum of 10 and 20")
        self.assertIn("30", response.answer)
    
    def test_routing_log(self):
        from bonus_reasoning_adapter.adapter import ReasoningAdapter
        adapter = ReasoningAdapter()
        adapter.route("test query 1")
        adapter.route("test query 2")
        stats = adapter.get_routing_stats()
        self.assertEqual(stats["total_queries"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
