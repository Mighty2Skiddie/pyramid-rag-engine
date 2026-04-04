"""
Bonus — Reasoning Adapter Interfaces

Defines abstract base classes for the plug-and-play reasoning system.
All domain-specific handlers implement the ReasoningHandler interface,
enabling runtime strategy selection based on query type.

This follows the Strategy Pattern: behavior is selected at runtime
based on input characteristics rather than being hardcoded.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class QueryType(Enum):
    """Supported query domains for routing."""
    MATH = "math"
    LEGAL = "legal"
    FACTUAL = "factual"
    CODE = "code"
    GENERAL = "general"


@dataclass
class ReasoningResponse:
    """
    Structured output from a reasoning handler.
    
    Attributes:
        answer:           Final answer text
        reasoning_trace:  Step-by-step reasoning (chain-of-thought)
        confidence:       Handler's confidence in the answer (0.0–1.0)
        handler_name:     Which handler produced this response
        query_type:       Detected query type
        metadata:         Additional handler-specific metadata
    """
    answer: str
    reasoning_trace: List[str]
    confidence: float
    handler_name: str
    query_type: QueryType
    metadata: Dict = field(default_factory=dict)


class ReasoningHandler(ABC):
    """
    Abstract base class for domain-specific reasoning handlers.
    
    Each handler implements a single method: handle(query) → Response.
    New domains are added by creating a new handler class and registering
    it in the HANDLER_REGISTRY — zero changes to existing code.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable handler name."""
        pass
    
    @property
    @abstractmethod
    def supported_type(self) -> QueryType:
        """The query type this handler processes."""
        pass
    
    @abstractmethod
    def handle(self, query: str, context: Optional[str] = None) -> ReasoningResponse:
        """
        Process a query and return a structured reasoning response.
        
        Args:
            query: User's input query
            context: Optional context from previous handler (for chaining)
        
        Returns:
            ReasoningResponse with answer, trace, and confidence
        """
        pass


class QueryClassifier(ABC):
    """
    Abstract base class for query type classifiers.
    
    The classifier is itself swappable: start with keyword rules
    (zero cost), upgrade to a fine-tuned classifier later. Both
    implement this same interface.
    """
    
    @abstractmethod
    def classify(self, query: str) -> tuple:
        """
        Classify a query into a QueryType with confidence.
        
        Args:
            query: User's input query
        
        Returns:
            Tuple of (QueryType, confidence_score)
        """
        pass
