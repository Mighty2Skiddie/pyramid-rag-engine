"""
Module 3 — Knowledge Pyramid Builder

Transforms each raw text chunk into a 4-layer hierarchical knowledge
representation. This is the core of the "Agentic Knowledge Distillation"
approach from the reference article.

Layer Architecture:
    L1 (Raw Text):          Ground truth — original chunk stored as-is
    L2 (Chunk Summary):     Rule-based extractive summary (first N sentences
                            + frequency-ranked sentence selection)
    L3 (Category Label):    Rule-based classification using keyword dictionaries
    L4 (Distilled Knowledge): TF-IDF keywords + hash-based mock embedding

Design Decision:
    All pyramid layers use rule-based/statistical methods instead of LLM calls.
    This keeps the system self-contained, reproducible, and free to run. The
    architecture is modular — replacing any layer with an LLM-based version
    (e.g., using FLAN-T5 for L2 summaries) requires changing only one function.
"""

from typing import Dict, List
from collections import Counter

import numpy as np

from part1_document_pipeline.models import Chunk, PyramidNode
from part1_document_pipeline.similarity import (
    build_tfidf_vectorizer,
    extract_top_keywords,
    generate_mock_embedding
)
from shared.text_utils import extract_sentences, tokenize_simple
from shared.config_manager import PyramidConfig
from shared.logger import get_logger

logger = get_logger(__name__)


# ────────────────────────────────────────────────────────────────
#  Main Pyramid Builder
# ────────────────────────────────────────────────────────────────

def build_pyramid(
    chunks: List[Chunk],
    config: PyramidConfig = None
) -> Dict[str, PyramidNode]:
    """
    Build the complete Knowledge Pyramid for a list of chunks.
    
    Each chunk is transformed into a PyramidNode containing all four
    layers. The TF-IDF vectorizer is fit once on the full corpus for
    consistent keyword extraction across all chunks.
    
    Args:
        chunks: List of Chunk dataclasses from the sliding window chunker
        config: Optional PyramidConfig (uses defaults if not provided)
    
    Returns:
        Dictionary mapping chunk_id → PyramidNode
    
    Example:
        >>> pyramid = build_pyramid(chunks)
        >>> node = pyramid["doc_abc_chunk_001"]
        >>> print(node.summary)         # L2
        >>> print(node.category)        # L3
        >>> print(node.keywords[:5])    # L4
    """
    if config is None:
        config = PyramidConfig()
    
    if not chunks:
        logger.warning("No chunks provided to pyramid builder")
        return {}
    
    # Pre-compute TF-IDF across all chunks for consistent keyword extraction
    all_texts = [chunk.text for chunk in chunks]
    vectorizer, _ = build_tfidf_vectorizer(all_texts)
    
    pyramid_index: Dict[str, PyramidNode] = {}
    
    for chunk in chunks:
        node = _build_single_node(chunk, vectorizer, config)
        pyramid_index[chunk.chunk_id] = node
    
    logger.info(
        f"Built pyramid: {len(pyramid_index)} nodes × 4 levels = "
        f"{len(pyramid_index) * 4} entries"
    )
    
    return pyramid_index


# ────────────────────────────────────────────────────────────────
#  Per-Chunk Node Builder
# ────────────────────────────────────────────────────────────────

def _build_single_node(
    chunk: Chunk,
    vectorizer,
    config: PyramidConfig
) -> PyramidNode:
    """Build a complete PyramidNode for a single chunk."""
    
    # L1: Raw Text (stored as-is)
    raw_text = chunk.text
    
    # L2: Chunk Summary
    summary = _generate_summary(raw_text, config.summary_sentence_count)
    
    # L3: Category Label
    category, confidence = _classify_category(raw_text, config.category_keywords)
    
    # L4: Distilled Knowledge (keywords + embedding)
    keywords = extract_top_keywords(
        raw_text, vectorizer, top_n=config.tfidf_top_n_keywords
    )
    embedding = generate_mock_embedding(raw_text, dim=config.embedding_dim)
    
    return PyramidNode(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        raw_text=raw_text,
        summary=summary,
        category=category,
        category_confidence=confidence,
        keywords=keywords,
        embedding=embedding
    )


# ────────────────────────────────────────────────────────────────
#  L2 — Summary Generation (Extractive)
# ────────────────────────────────────────────────────────────────

def _generate_summary(text: str, num_sentences: int = 2) -> str:
    """
    Generate an extractive summary using sentence scoring.
    
    Strategy:
        1. Split text into sentences
        2. Score each sentence by word frequency (simplified TextRank)
        3. Select the top-N highest-scoring sentences
        4. Return them in original order for coherence
    
    In production, replace with an LLM-based abstractive summarizer
    (e.g., FLAN-T5 or GPT-4 API call).
    
    Args:
        text: Raw chunk text
        num_sentences: Number of sentences to include in summary
    
    Returns:
        Summary string
    """
    sentences = extract_sentences(text)
    
    if not sentences:
        return text[:200] if text else ""
    
    if len(sentences) <= num_sentences:
        return " ".join(sentences)
    
    # Score sentences by word frequency (simplified TextRank)
    # Step 1: Build word frequency from the full text
    words = tokenize_simple(text)
    word_freq = Counter(words)
    
    # Step 2: Score each sentence by sum of word frequencies
    sentence_scores = []
    for i, sentence in enumerate(sentences):
        s_words = tokenize_simple(sentence)
        if not s_words:
            sentence_scores.append((i, 0.0))
            continue
        
        score = sum(word_freq.get(w, 0) for w in s_words) / len(s_words)
        sentence_scores.append((i, score))
    
    # Step 3: Select top-N sentences by score
    ranked = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
    top_indices = sorted([idx for idx, _ in ranked[:num_sentences]])
    
    # Step 4: Return in original order for coherence
    summary = " ".join(sentences[i] for i in top_indices)
    
    return summary


# ────────────────────────────────────────────────────────────────
#  L3 — Category Classification (Rule-Based)
# ────────────────────────────────────────────────────────────────

def _classify_category(
    text: str,
    category_keywords: Dict[str, List[str]]
) -> tuple:
    """
    Classify text into a category using keyword matching.
    
    Strategy:
        1. Tokenize the text into lowercase words
        2. Count matches against each category's keyword list
        3. The category with the most matches wins
        4. Confidence = matched_count / total_keywords_checked
    
    Args:
        text: Raw chunk text
        category_keywords: Dict mapping category name → keyword list
    
    Returns:
        Tuple of (category_label, confidence_score)
    """
    text_lower = text.lower()
    text_words = set(text_lower.split())
    
    category_scores: Dict[str, int] = {}
    
    for category, keywords in category_keywords.items():
        # Count how many category keywords appear in the text
        matches = sum(1 for kw in keywords if kw in text_words or kw in text_lower)
        category_scores[category] = matches
    
    # Find the best category
    if not category_scores or max(category_scores.values()) == 0:
        return "general", 0.0
    
    best_category = max(category_scores, key=category_scores.get)
    best_score = category_scores[best_category]
    
    # Confidence: ratio of matched keywords to total keywords in that category
    total_keywords = len(category_keywords.get(best_category, []))
    confidence = best_score / total_keywords if total_keywords > 0 else 0.0
    
    # If confidence is very low, default to "general"
    if confidence < 0.05:
        return "general", confidence
    
    return best_category, round(min(confidence, 1.0), 3)
