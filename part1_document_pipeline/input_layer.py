"""
Module 1 — Input Layer

Accepts raw documents (string, .txt file, .pdf file) and normalizes them
into a consistent Document dataclass for downstream processing.

Design Decision:
    The input layer is intentionally thin and format-agnostic. Adding
    support for new formats (DOCX, HTML, etc.) requires only adding a
    new loader function here — zero changes downstream.
"""

import os
import hashlib
from pathlib import Path
from typing import Union

from part1_document_pipeline.models import Document
from shared.text_utils import normalize_text, estimate_token_count
from shared.logger import get_logger

logger = get_logger(__name__)


def load_document(source: Union[str, Path]) -> Document:
    """
    Load and normalize a document from various input sources.
    
    Accepts:
        - A plain string (treated as raw text content)
        - A path to a .txt file
        - A path to a .pdf file (requires PyMuPDF)
    
    Args:
        source: Raw text string, or file path (str or Path)
    
    Returns:
        Document dataclass with normalized text and metadata
    
    Raises:
        FileNotFoundError: If file path doesn't exist
        ValueError: If document is empty after normalization
    """
    source_str = str(source)
    
    # Determine input type and extract raw text
    if os.path.isfile(source_str):
        raw_text = _load_from_file(source_str)
        source_label = source_str
        logger.info(f"Loaded file: {source_str}")
    else:
        # Treat as raw text string
        raw_text = source_str
        source_label = "string_input"
        logger.info(f"Received raw text input ({len(raw_text)} chars)")
    
    # Normalize the text
    cleaned_text = normalize_text(raw_text)
    
    if not cleaned_text:
        raise ValueError("Document is empty after normalization.")
    
    # Generate deterministic doc_id from content hash
    content_hash = hashlib.md5(cleaned_text.encode("utf-8")).hexdigest()[:8]
    doc_id = f"doc_{content_hash}"
    
    char_count = len(cleaned_text)
    token_count = estimate_token_count(cleaned_text)
    
    logger.info(f"Document '{doc_id}': {char_count} chars, ~{token_count} tokens")
    
    return Document(
        doc_id=doc_id,
        text=cleaned_text,
        source=source_label,
        char_count=char_count,
        token_count=token_count
    )


def _load_from_file(file_path: str) -> str:
    """
    Load text content from a file based on its extension.
    
    Supports:
        - .txt: Plain text reading with UTF-8 encoding
        - .pdf: Text extraction via PyMuPDF (fitz)
    
    Args:
        file_path: Absolute or relative path to the file
    
    Returns:
        Extracted text string
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        return _load_pdf(file_path)
    elif ext in (".txt", ".md", ".text"):
        return _load_text(file_path)
    else:
        # Default: try reading as text
        logger.warning(f"Unknown extension '{ext}', attempting plain text read")
        return _load_text(file_path)


def _load_text(file_path: str) -> str:
    """Read a plain text file with UTF-8 encoding."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _load_pdf(file_path: str) -> str:
    """
    Extract text from a PDF using PyMuPDF (fitz).
    
    Concatenates text from all pages with page boundary markers.
    Falls back gracefully if PyMuPDF is not installed.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error(
            "PyMuPDF not installed. Install with: pip install PyMuPDF"
        )
        raise ImportError(
            "PDF support requires PyMuPDF. Install: pip install PyMuPDF"
        )
    
    doc = fitz.open(file_path)
    pages = []
    
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        if page_text.strip():
            pages.append(page_text)
    
    doc.close()
    
    # Join pages with double newline as boundary marker
    full_text = "\n\n".join(pages)
    logger.info(f"Extracted {len(pages)} pages from PDF: {file_path}")
    
    return full_text
