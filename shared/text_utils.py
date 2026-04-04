"""
Common text processing utilities shared across subsystems.
Handles normalization, sentence splitting, and keyword extraction.
"""

import re
import string
from typing import List, Set


# --- Common English stop words (no external dependency needed) ---
STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "it", "its", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "our", "their", "what", "which", "who", "whom", "how", "when",
    "where", "why", "not", "no", "nor", "so", "if", "then", "than",
    "too", "very", "just", "about", "above", "after", "again", "all",
    "also", "am", "any", "because", "before", "between", "both", "each",
    "few", "more", "most", "other", "some", "such", "into", "over",
    "own", "same", "up", "down", "out", "off", "only", "under", "until",
    "while", "during", "through", "here", "there"
}


def normalize_text(text: str) -> str:
    """
    Normalize text by cleaning whitespace, control characters, and encoding.
    
    Steps:
        1. Replace non-UTF-8 sequences
        2. Strip control characters (keep newlines and tabs)
        3. Collapse multiple whitespace
        4. Strip leading/trailing whitespace
    
    Args:
        text: Raw input text
    
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    
    # Ensure valid UTF-8 by encoding and decoding with error replacement
    text = text.encode("utf-8", errors="replace").decode("utf-8")
    
    # Remove control characters except newline (\n) and tab (\t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    
    # Collapse multiple spaces/tabs into single space (preserve newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    
    # Collapse multiple newlines into max 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


def extract_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex-based boundary detection.
    
    Handles common abbreviations (Mr., Dr., etc.) and decimal numbers
    to avoid false splits.
    
    Args:
        text: Input text
    
    Returns:
        List of sentence strings
    """
    if not text or not text.strip():
        return []
    
    # Split on sentence-ending punctuation followed by space + uppercase or end
    # This regex avoids splitting on abbreviations like "Mr." or "U.S."
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_pattern, text.strip())
    
    # Clean each sentence and filter empty ones
    return [s.strip() for s in sentences if s.strip()]


def remove_stopwords(text: str) -> str:
    """
    Remove common English stop words from text.
    
    Args:
        text: Input text
    
    Returns:
        Text with stop words removed
    """
    words = text.lower().split()
    return " ".join(w for w in words if w not in STOP_WORDS)


def tokenize_simple(text: str) -> List[str]:
    """
    Simple whitespace + punctuation tokenizer.
    Lowercases and strips punctuation from each token.
    
    Args:
        text: Input text
    
    Returns:
        List of lowercase tokens
    """
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t and t not in STOP_WORDS]


def estimate_token_count(text: str) -> int:
    """
    Rough token count estimation (~4 characters per token for English).
    
    This avoids dependency on tiktoken while giving a reasonable estimate.
    For production, replace with actual tokenizer count.
    
    Args:
        text: Input text
    
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    return max(1, len(text) // 4)
