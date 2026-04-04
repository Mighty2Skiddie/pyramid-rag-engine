"""
Configuration manager using dataclasses and optional YAML support.
Provides typed configuration for all system components.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ChunkerConfig:
    """Configuration for the sliding window chunker."""
    window_size_chars: int = 2000          # ~2 pages worth of text
    overlap_ratio: float = 0.15            # 15% overlap between windows
    min_chunk_size: int = 100              # Minimum chars for a valid chunk
    use_token_based: bool = False          # Fallback: char-based windowing


@dataclass
class PyramidConfig:
    """Configuration for the Knowledge Pyramid builder."""
    summary_sentence_count: int = 2        # Sentences for L2 summary
    tfidf_top_n_keywords: int = 10         # Top keywords for L4
    embedding_dim: int = 128               # Mock embedding vector dimension
    
    # Category keyword dictionaries for L3 rule-based classification
    category_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        "finance": [
            "revenue", "profit", "loss", "market", "stock", "investment",
            "capital", "dividend", "earnings", "fiscal", "budget", "cost",
            "price", "valuation", "equity", "debt", "interest", "tax",
            "financial", "banking", "fund", "asset", "liability", "margin"
        ],
        "legal": [
            "section", "clause", "liability", "agreement", "contract",
            "regulation", "compliance", "statute", "jurisdiction", "court",
            "plaintiff", "defendant", "attorney", "legal", "law", "rights",
            "obligation", "warranty", "indemnity", "arbitration", "breach",
            "damages", "tort", "provision", "amendment"
        ],
        "technical": [
            "function", "algorithm", "model", "data", "system", "software",
            "hardware", "api", "database", "server", "network", "code",
            "deploy", "architecture", "framework", "protocol", "compute",
            "memory", "processor", "interface", "module", "pipeline",
            "parameter", "optimization", "latency", "throughput"
        ],
        "medical": [
            "patient", "diagnosis", "treatment", "clinical", "symptom",
            "therapy", "disease", "medication", "hospital", "surgery",
            "dosage", "physician", "health", "medical", "prognosis"
        ]
    })


@dataclass
class RetrieverConfig:
    """Configuration for the semantic retriever."""
    top_k: int = 3                         # Number of results to return
    
    # Weights for each pyramid level (must sum to ~1.0)
    # L4 (distilled knowledge) weighted highest per design rationale
    level_weights: Dict[str, float] = field(default_factory=lambda: {
        "L1": 0.15,   # Raw text — broad but noisy
        "L2": 0.30,   # Summary — balanced precision
        "L3": 0.10,   # Category — coarse signal
        "L4": 0.45    # Distilled keywords + embedding — highest signal
    })
    
    min_confidence_threshold: float = 0.05  # Minimum score to include result


@dataclass
class SystemConfig:
    """Top-level configuration combining all subsystem configs."""
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    pyramid: PyramidConfig = field(default_factory=PyramidConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)


def load_config() -> SystemConfig:
    """
    Load system configuration, with environment variable overrides.

    Reads dataclass defaults first, then applies any environment variable
    overrides — useful for tuning behavior in CI, staging, or production
    without changing code.

    Supported overrides:
        CHUNKER_WINDOW_SIZE   — int, default 2000
        CHUNKER_OVERLAP_RATIO — float, default 0.15
        RETRIEVER_TOP_K       — int, default 3
        RETRIEVER_MIN_CONF    — float, default 0.05
        PYRAMID_SUMMARY_SENTS — int, default 2

    Returns:
        SystemConfig instance with all subsystem configurations
    """
    import os

    chunker = ChunkerConfig(
        window_size_chars=int(os.environ.get("CHUNKER_WINDOW_SIZE", 2000)),
        overlap_ratio=float(os.environ.get("CHUNKER_OVERLAP_RATIO", 0.15)),
    )

    pyramid = PyramidConfig(
        summary_sentence_count=int(os.environ.get("PYRAMID_SUMMARY_SENTS", 2)),
    )

    retriever = RetrieverConfig(
        top_k=int(os.environ.get("RETRIEVER_TOP_K", 3)),
        min_confidence_threshold=float(os.environ.get("RETRIEVER_MIN_CONF", 0.05)),
    )

    return SystemConfig(chunker=chunker, pyramid=pyramid, retriever=retriever)
