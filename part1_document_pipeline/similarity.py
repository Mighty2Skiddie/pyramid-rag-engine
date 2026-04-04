"""
Similarity computation utilities for the Knowledge Pyramid retriever.

Provides multiple similarity methods to match queries against different
pyramid levels:
    - TF-IDF cosine similarity (L1 raw text, L2 summaries)
    - Fuzzy string matching via rapidfuzz (L2 summaries)
    - Keyword overlap Jaccard score (L4 keywords)
    - Vector cosine similarity (L4 embeddings)
    - Mock embedding generation (deterministic hash-based)

Design Decision:
    scikit-learn TF-IDF is used over sentence-transformers because the
    assignment scope doesn't require deep semantic embeddings. The 
    architecture is designed so swapping in real embeddings requires
    changing only generate_embedding() below.
"""

import hashlib
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from shared.logger import get_logger

logger = get_logger(__name__)


# ────────────────────────────────────────────────────────────────
#  TF-IDF Based Similarity
# ────────────────────────────────────────────────────────────────

def compute_tfidf_similarity(query: str, documents: List[str]) -> List[float]:
    """
    Compute TF-IDF cosine similarity between a query and a list of documents.
    
    The vectorizer is fit on the combined corpus (query + documents) to
    create a shared vocabulary space.
    
    Args:
        query: Query text string
        documents: List of document text strings
    
    Returns:
        List of similarity scores (0.0–1.0) for each document
    """
    if not query or not documents:
        return [0.0] * len(documents)
    
    # Fit vectorizer on full corpus for shared vocabulary
    corpus = [query] + documents
    
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2)  # Unigrams + bigrams for better matching
        )
        tfidf_matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        # All documents are empty or contain only stop words
        return [0.0] * len(documents)
    
    # Query vector is the first row, documents are the rest
    query_vec = tfidf_matrix[0:1]
    doc_vecs = tfidf_matrix[1:]
    
    # Compute cosine similarity
    similarities = sklearn_cosine(query_vec, doc_vecs).flatten()
    
    return similarities.tolist()


def build_tfidf_vectorizer(texts: List[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    """
    Build a TF-IDF vectorizer fit on a corpus and return both the
    vectorizer and the document-term matrix.
    
    Used during pyramid building to pre-compute TF-IDF representations
    for all chunks. The vectorizer can then be reused at query time.
    
    Args:
        texts: List of text strings (all chunks)
    
    Returns:
        Tuple of (fitted TfidfVectorizer, document-term matrix)
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2)
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError:
        logger.warning("TF-IDF vectorizer received empty corpus")
        return vectorizer, None
    
    return vectorizer, tfidf_matrix


def extract_top_keywords(
    text: str,
    vectorizer: TfidfVectorizer,
    top_n: int = 10
) -> List[str]:
    """
    Extract top-N keywords from text using a pre-fit TF-IDF vectorizer.
    
    Args:
        text: Input text
        vectorizer: Pre-fit TfidfVectorizer
        top_n: Number of top keywords to extract
    
    Returns:
        List of top keywords sorted by TF-IDF score
    """
    try:
        tfidf_vector = vectorizer.transform([text])
    except Exception:
        return []
    
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_vector.toarray().flatten()
    
    # Get indices of top-N scores
    top_indices = scores.argsort()[-top_n:][::-1]
    
    keywords = [
        feature_names[i] for i in top_indices if scores[i] > 0
    ]
    
    return keywords


# ────────────────────────────────────────────────────────────────
#  Fuzzy String Matching
# ────────────────────────────────────────────────────────────────

def fuzzy_match_score(query: str, text: str) -> float:
    """
    Compute fuzzy string similarity using rapidfuzz.
    
    Uses token_set_ratio which handles word order differences and
    partial matches better than simple ratio for natural language.
    
    Args:
        query: Query string
        text: Target text to compare against
    
    Returns:
        Similarity score (0.0–1.0)
    """
    if not query or not text:
        return 0.0
    
    try:
        from rapidfuzz import fuzz
        # token_set_ratio is best for comparing queries against summaries
        # because it handles different word orders and extra words
        score = fuzz.token_set_ratio(query.lower(), text.lower())
        return score / 100.0  # Normalize to 0.0–1.0
    except ImportError:
        logger.warning("rapidfuzz not installed, falling back to basic matching")
        return _basic_token_overlap(query, text)


def _basic_token_overlap(query: str, text: str) -> float:
    """Fallback: simple token overlap ratio if rapidfuzz is unavailable."""
    q_tokens = set(query.lower().split())
    t_tokens = set(text.lower().split())
    
    if not q_tokens or not t_tokens:
        return 0.0
    
    overlap = len(q_tokens & t_tokens)
    return overlap / max(len(q_tokens), len(t_tokens))


# ────────────────────────────────────────────────────────────────
#  Keyword Overlap (Jaccard)
# ────────────────────────────────────────────────────────────────

def keyword_jaccard_score(
    query_keywords: List[str],
    chunk_keywords: List[str]
) -> float:
    """
    Compute Jaccard similarity between two sets of keywords.
    
    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    
    Args:
        query_keywords: Keywords extracted from the query
        chunk_keywords: Keywords from a pyramid L4 node
    
    Returns:
        Jaccard similarity score (0.0–1.0)
    """
    if not query_keywords or not chunk_keywords:
        return 0.0
    
    set_a = set(k.lower() for k in query_keywords)
    set_b = set(k.lower() for k in chunk_keywords)
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return intersection / union if union > 0 else 0.0


# ────────────────────────────────────────────────────────────────
#  Vector Cosine Similarity
# ────────────────────────────────────────────────────────────────

def vector_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector (1D numpy array)
        vec2: Second vector (1D numpy array)
    
    Returns:
        Cosine similarity score (-1.0 to 1.0, typically 0.0–1.0 for
        non-negative vectors)
    """
    if vec1 is None or vec2 is None:
        return 0.0
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


# ────────────────────────────────────────────────────────────────
#  Mock Embedding Generation
# ────────────────────────────────────────────────────────────────

def generate_mock_embedding(text: str, dim: int = 128) -> np.ndarray:
    """
    Generate a deterministic mock embedding vector by hashing text content.
    
    This produces a normalized float vector that is consistent for the
    same input text. In production, replace this with a real embedding
    model (e.g., sentence-transformers all-MiniLM-L6-v2).
    
    The hash-seeded RNG approach ensures:
        - Deterministic output (same text → same vector)
        - Well-distributed float values (no overflow)
        - Non-zero, unit-normalized vectors for non-empty text
    
    Args:
        text: Input text to embed
        dim: Embedding dimension (default: 128)
    
    Returns:
        Normalized numpy array of shape (dim,)
    """
    if not text:
        return np.zeros(dim)
    
    # Create a deterministic seed from text content via hash
    text_lower = text.lower().strip()
    hash_int = int(hashlib.sha256(text_lower.encode("utf-8")).hexdigest(), 16)
    
    # Use the hash as a seed for a random number generator
    # This produces well-behaved float values in [-1, 1]
    rng = np.random.RandomState(hash_int % (2**31))
    raw = rng.randn(dim).astype(np.float32)
    
    # Normalize to unit vector
    norm = np.linalg.norm(raw)
    if norm > 0:
        raw = raw / norm
    
    return raw
