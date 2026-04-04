"""
Module 2 — Sliding Window Chunker

Splits a normalized document into overlapping text windows that simulate
"2-page" segments. Overlap ensures no semantic context is lost at
window boundaries.

Design Decision:
    Character-based windowing is the default because it requires no
    external tokenizer dependency. For LLM-adjacent workflows where
    tokenization is the true unit of model context, token-based
    windowing can be enabled via config.
"""

import hashlib
from typing import List

from part1_document_pipeline.models import Document, Chunk
from shared.text_utils import estimate_token_count
from shared.config_manager import ChunkerConfig
from shared.logger import get_logger

logger = get_logger(__name__)


def chunk_document(
    document: Document,
    config: ChunkerConfig = None
) -> List[Chunk]:
    """
    Split a document into overlapping sliding window chunks.
    
    The chunker uses a configurable window size and overlap ratio:
    - Window size: ~2000 chars (approximately 2 pages of text)
    - Overlap: 15% of window size to preserve cross-boundary context
    
    Args:
        document: Normalized Document dataclass
        config: Optional ChunkerConfig (uses defaults if not provided)
    
    Returns:
        List of Chunk dataclasses with position metadata
    
    Edge Cases Handled:
        - Empty document → returns empty list
        - Document shorter than window → single chunk
        - Duplicate chunks → deduplicated by content hash
    """
    if config is None:
        config = ChunkerConfig()
    
    text = document.text
    
    # Edge case: empty document
    if not text or len(text.strip()) == 0:
        logger.warning(f"Document '{document.doc_id}' is empty, returning no chunks")
        return []
    
    window_size = config.window_size_chars
    overlap = int(window_size * config.overlap_ratio)
    step_size = window_size - overlap  # How far to advance each window
    
    # Edge case: document shorter than one window
    if len(text) <= window_size:
        logger.info(f"Document fits in single window ({len(text)} <= {window_size} chars)")
        chunk = Chunk(
            chunk_id=f"{document.doc_id}_chunk_001",
            doc_id=document.doc_id,
            text=text,
            start_char=0,
            end_char=len(text),
            token_count=estimate_token_count(text)
        )
        return [chunk]
    
    chunks: List[Chunk] = []
    seen_hashes = set()  # For content deduplication
    chunk_num = 0
    pos = 0
    
    while pos < len(text):
        # Extract the window
        end_pos = min(pos + window_size, len(text))
        chunk_text = text[pos:end_pos]
        
        # Skip chunks that are too small (unless it's the last one)
        if len(chunk_text.strip()) < config.min_chunk_size and pos > 0:
            logger.debug(f"Skipping small trailing chunk ({len(chunk_text)} chars)")
            break
        
        # Deduplicate by content hash
        content_hash = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()
        if content_hash in seen_hashes:
            logger.debug(f"Skipping duplicate chunk at position {pos}")
            pos += step_size
            continue
        seen_hashes.add(content_hash)
        
        chunk_num += 1
        chunk_id = f"{document.doc_id}_chunk_{chunk_num:03d}"
        
        chunk = Chunk(
            chunk_id=chunk_id,
            doc_id=document.doc_id,
            text=chunk_text,
            start_char=pos,
            end_char=end_pos,
            token_count=estimate_token_count(chunk_text)
        )
        chunks.append(chunk)
        
        # Advance the window
        pos += step_size
        
        # If we've reached the end, stop
        if end_pos >= len(text):
            break
    
    logger.info(
        f"Created {len(chunks)} chunks from '{document.doc_id}' "
        f"(window={window_size}, overlap={overlap}, step={step_size})"
    )
    
    return chunks
