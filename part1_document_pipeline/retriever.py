"""
Module 4 — Query Interface & Semantic Retriever

Accepts natural language queries and retrieves the most relevant content
from any layer of the Knowledge Pyramid using multi-level similarity.

Retrieval Strategy:
    The retriever computes similarity at each pyramid level independently,
    then aggregates scores using configurable level weights:
        L1 (Raw Text):     TF-IDF cosine — broad lexical matching
        L2 (Summary):      Fuzzy match — flexible sentence-level matching
        L3 (Category):     Label matching — coarse domain filter
        L4 (Distilled):    Keyword Jaccard + vector cosine — dense matching
    
    Scores are weighted and combined. The top-K candidates are returned
    with full provenance metadata showing which level contributed most.

Design Decision:
    Aggregated multi-level scoring outperforms single-level retrieval because
    different query types benefit from different abstraction levels. A factual
    query about "revenue growth" may match best at L4 (keywords), while a
    conceptual query about "business strategy" may match best at L2 (summary).
"""

from typing import Dict, List, Optional

import numpy as np

from part1_document_pipeline.models import PyramidNode, QueryResult
from part1_document_pipeline.similarity import (
    compute_tfidf_similarity,
    fuzzy_match_score,
    keyword_jaccard_score,
    vector_cosine_similarity,
    generate_mock_embedding,
    extract_top_keywords,
    build_tfidf_vectorizer
)
from shared.text_utils import tokenize_simple
from shared.config_manager import RetrieverConfig
from shared.logger import get_logger

logger = get_logger(__name__)


def query_pyramid(
    query_text: str,
    pyramid_index: Dict[str, PyramidNode],
    config: RetrieverConfig = None
) -> List[QueryResult]:
    """
    Retrieve the most relevant chunks from the Knowledge Pyramid.
    
    Process:
        1. Normalize the query and extract keywords/embedding
        2. Compute similarity at each pyramid level for every chunk
        3. Aggregate level scores using configurable weights
        4. Rank and return top-K results with metadata
    
    Args:
        query_text: Natural language query string
        pyramid_index: Dictionary of chunk_id → PyramidNode
        config: Optional RetrieverConfig (uses defaults if not provided)
    
    Returns:
        List of QueryResult objects, sorted by score descending
    
    Example:
        >>> results = query_pyramid("What are the revenue projections?", pyramid)
        >>> for r in results:
        ...     print(f"[{r.best_level}] {r.chunk_id}: {r.score:.3f}")
    """
    if config is None:
        config = RetrieverConfig()
    
    if not query_text or not query_text.strip():
        logger.warning("Empty query received")
        return []
    
    if not pyramid_index:
        logger.warning("Empty pyramid index — no documents loaded")
        return []
    
    query_text = query_text.strip()
    logger.info(f"Processing query: '{query_text[:80]}...'")
    
    # ── Step 1: Prepare query representations ──
    query_keywords = tokenize_simple(query_text)
    query_embedding = generate_mock_embedding(query_text)
    
    # Build TF-IDF on the corpus of raw texts for L1 similarity
    chunk_ids = list(pyramid_index.keys())
    nodes = [pyramid_index[cid] for cid in chunk_ids]
    raw_texts = [n.raw_text for n in nodes]
    summaries = [n.summary for n in nodes]
    
    # ── Step 2: Compute L1 scores (TF-IDF cosine on raw text) ──
    l1_scores = compute_tfidf_similarity(query_text, raw_texts)
    
    # ── Step 3: Compute L2 scores (fuzzy match on summaries) ──
    l2_scores = [fuzzy_match_score(query_text, s) for s in summaries]
    
    # ── Step 4: Compute L3 scores (category label matching) ──
    # Infer the query's likely category and compare
    query_category = _infer_query_category(query_text)
    l3_scores = [
        1.0 if n.category == query_category and query_category != "general" else
        0.3 if n.category == query_category else
        0.0
        for n in nodes
    ]
    
    # ── Step 5: Compute L4 scores (keyword Jaccard + vector cosine) ──
    l4_scores = []
    for node in nodes:
        jaccard = keyword_jaccard_score(query_keywords, node.keywords)
        vec_sim = vector_cosine_similarity(query_embedding, node.embedding)
        # Combine keyword and vector scores (weighted average)
        l4 = 0.6 * jaccard + 0.4 * max(0, vec_sim)
        l4_scores.append(l4)
    
    # ── Step 6: Aggregate scores with level weights ──
    weights = config.level_weights
    results: List[QueryResult] = []
    
    for i, chunk_id in enumerate(chunk_ids):
        node = nodes[i]
        
        level_scores = {
            "L1": l1_scores[i],
            "L2": l2_scores[i],
            "L3": l3_scores[i],
            "L4": l4_scores[i]
        }
        
        # Weighted aggregate
        total_score = (
            weights["L1"] * level_scores["L1"] +
            weights["L2"] * level_scores["L2"] +
            weights["L3"] * level_scores["L3"] +
            weights["L4"] * level_scores["L4"]
        )
        
        # Determine which level contributed most
        best_level = max(level_scores, key=level_scores.get)
        
        # Filter by minimum confidence threshold
        if total_score < config.min_confidence_threshold:
            continue
        
        results.append(QueryResult(
            chunk_id=chunk_id,
            score=round(total_score, 4),
            best_level=best_level,
            level_scores={k: round(v, 4) for k, v in level_scores.items()},
            raw_text=node.raw_text[:500],  # Truncated for display
            summary=node.summary,
            category=node.category,
            keywords=node.keywords[:5]     # Top 5 for display
        ))
    
    # Sort by score descending, return top-K
    results.sort(key=lambda r: r.score, reverse=True)
    top_results = results[:config.top_k]
    
    if top_results:
        logger.info(
            f"Retrieved {len(top_results)} results "
            f"(top score: {top_results[0].score:.3f})"
        )
    else:
        logger.info("No results above confidence threshold")
    
    return top_results


def _infer_query_category(query: str) -> str:
    """
    Infer the likely category/domain of a query using keyword heuristics.
    
    This mirrors the L3 classification logic but applied to the query
    to enable category-level matching.
    
    Args:
        query: Query text
    
    Returns:
        Inferred category string
    """
    query_lower = query.lower()
    
    category_signals = {
        "finance": ["revenue", "profit", "market", "stock", "financial",
                     "earnings", "cost", "price", "investment", "budget"],
        "legal": ["law", "legal", "clause", "contract", "regulation",
                  "court", "liability", "rights", "compliance", "statute"],
        "technical": ["algorithm", "function", "code", "system", "data",
                      "model", "api", "software", "architecture", "deploy"],
        "medical": ["patient", "treatment", "diagnosis", "clinical",
                    "disease", "symptom", "therapy", "medical", "health"]
    }
    
    scores = {}
    for cat, keywords in category_signals.items():
        scores[cat] = sum(1 for kw in keywords if kw in query_lower)
    
    best_cat = max(scores, key=scores.get)
    return best_cat if scores[best_cat] > 0 else "general"


# ────────────────────────────────────────────────────────────────
#  Convenience: Full Pipeline Runner
# ────────────────────────────────────────────────────────────────

def format_results(results: List[QueryResult], query: str) -> str:
    """
    Format query results into a readable string output.
    
    Args:
        results: List of QueryResult objects
        query: Original query string
    
    Returns:
        Formatted string for display
    """
    lines = [
        f"\n{'='*70}",
        f"  QUERY: \"{query}\"",
        f"{'='*70}"
    ]
    
    if not results:
        lines.append("  No relevant content found.")
        return "\n".join(lines)
    
    for rank, r in enumerate(results, 1):
        lines.extend([
            f"\n  Rank {rank} | Score: {r.score:.4f} | "
            f"Best Level: {r.best_level} | Chunk: {r.chunk_id}",
            f"  {'-'*60}",
            f"  Category: {r.category}",
            f"  Keywords: {', '.join(r.keywords)}",
            f"  Summary: {r.summary[:150]}...",
            f"  Level Scores: {r.level_scores}",
        ])
    
    lines.append(f"\n{'='*70}\n")
    return "\n".join(lines)
