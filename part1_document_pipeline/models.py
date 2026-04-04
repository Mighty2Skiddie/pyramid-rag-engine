"""
Data models for the Document Ingestion Pipeline.

All core data structures are defined as Python dataclasses for:
- Type safety and IDE autocompletion
- Immutable-by-default design (frozen where appropriate)
- Clean serialization and repr
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


# ────────────────────────────────────────────────────────────────
#  Input Layer Models
# ────────────────────────────────────────────────────────────────

@dataclass
class Document:
    """
    Represents a normalized input document ready for chunking.
    
    Attributes:
        doc_id:      Unique identifier for this document session
        text:        Cleaned, UTF-8 normalized text content
        source:      Original source (file path, URL, or "string")
        char_count:  Number of characters in the text
        token_count: Estimated token count (~4 chars/token)
    """
    doc_id: str
    text: str
    source: str
    char_count: int
    token_count: int


# ────────────────────────────────────────────────────────────────
#  Chunker Models
# ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    A single sliding window text chunk with position metadata.
    
    Attributes:
        chunk_id:    Unique ID within the document (e.g., "chunk_001")
        doc_id:      Parent document ID for traceability
        text:        The raw text content of this window
        start_char:  Starting character position in the original document
        end_char:    Ending character position in the original document
        token_count: Estimated token count for this chunk
    """
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    token_count: int


# ────────────────────────────────────────────────────────────────
#  Knowledge Pyramid Models
# ────────────────────────────────────────────────────────────────

@dataclass
class PyramidNode:
    """
    A single chunk's complete 4-layer Knowledge Pyramid representation.
    
    The pyramid transforms raw text into progressively more distilled
    representations, enabling retrieval at different levels of abstraction.
    
    Layers:
        L1 (Raw Text):      Original chunk text — ground truth
        L2 (Summary):       Condensed version — key sentences
        L3 (Category):      Theme label — coarse classification
        L4 (Distilled):     Keywords + embedding — dense representation
    """
    chunk_id: str
    doc_id: str
    
    # L1 — Raw Text (ground truth)
    raw_text: str
    
    # L2 — Chunk Summary
    summary: str
    
    # L3 — Category / Theme Label
    category: str
    category_confidence: float  # 0.0–1.0 confidence in the classification
    
    # L4 — Distilled Knowledge
    keywords: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None  # Mock embedding vector


# ────────────────────────────────────────────────────────────────
#  Retriever Models
# ────────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """
    A single retrieval result from the Knowledge Pyramid.
    
    Attributes:
        chunk_id:       ID of the matched chunk
        score:          Aggregated similarity score (0.0–1.0)
        best_level:     Which pyramid level produced the best match
        level_scores:   Per-level similarity breakdown
        raw_text:       Original chunk text for display
        summary:        L2 summary of the matched chunk
        category:       L3 category label
        keywords:       L4 distilled keywords
    """
    chunk_id: str
    score: float
    best_level: str
    level_scores: Dict[str, float]
    raw_text: str
    summary: str
    category: str
    keywords: List[str]


@dataclass
class SessionState:
    """
    Tracks the current pipeline session state.
    
    Attributes:
        doc_id:         Currently loaded document ID (None if no doc loaded)
        chunk_count:    Number of chunks in the current document
        pyramid_built:  Whether the pyramid has been built
        query_history:  List of past queries in this session
    """
    doc_id: Optional[str] = None
    chunk_count: int = 0
    pyramid_built: bool = False
    query_history: List[str] = field(default_factory=list)
