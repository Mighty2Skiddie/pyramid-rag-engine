"""
Bonus — Query Type Classifier

Detects the domain of an incoming query using keyword heuristics.
Designed to be swappable with an ML-based classifier without
changing any downstream code.

Current Implementation: Rule-based keyword matching (zero cost, no ML)
Production Upgrade:     Fine-tuned text classifier (e.g., DistilBERT)
"""

from typing import Dict, List, Tuple

from bonus_reasoning_adapter.interfaces import QueryType, QueryClassifier


class KeywordQueryClassifier(QueryClassifier):
    """
    Rule-based query classifier using keyword pattern matching.
    
    Each query type has a set of signal words. The type with the
    most keyword matches wins. Confidence is proportional to the
    match ratio.
    
    This is intentionally simple — it demonstrates the routing
    architecture without requiring ML infrastructure. The classifier
    can be upgraded to an ML model by implementing the same
    QueryClassifier interface.
    """
    
    # Keyword dictionaries for each query type
    TYPE_KEYWORDS: Dict[QueryType, List[str]] = {
        QueryType.MATH: [
            "calculate", "compute", "sum", "add", "subtract", "multiply",
            "divide", "percentage", "average", "total", "how many",
            "how much", "equation", "solve", "math", "number", "formula",
            "ratio", "proportion", "probability", "area", "volume",
            "distance", "speed", "rate", "cost per", "price per",
            "if each", "how old", "how far"
        ],
        QueryType.LEGAL: [
            "law", "legal", "regulation", "statute", "clause", "contract",
            "liability", "rights", "court", "judge", "attorney", "sue",
            "compliance", "violation", "penalty", "amendment", "act",
            "ordinance", "precedent", "ruling", "jurisdiction", "appeal",
            "warrant", "copyright", "patent infringement"
        ],
        QueryType.CODE: [
            "code", "function", "class", "bug", "error", "debug",
            "implement", "algorithm", "python", "javascript", "java",
            "compile", "runtime", "syntax", "variable", "loop", "array",
            "recursion", "api", "endpoint", "database query", "sql",
            "regex", "git", "deploy", "refactor"
        ],
        QueryType.FACTUAL: [
            "what is", "who is", "where is", "when did", "define",
            "explain", "describe", "history of", "origin of", "meaning of",
            "fact", "true or false", "capital of", "population of",
            "founded", "invented", "discovered", "located"
        ],
        # GENERAL has no specific keywords — it's the default fallback
    }
    
    def classify(self, query: str) -> Tuple[QueryType, float]:
        """
        Classify query using keyword matching.
        
        Args:
            query: User's input query
        
        Returns:
            Tuple of (QueryType, confidence) where confidence ∈ [0.0, 1.0]
        """
        if not query or not query.strip():
            return QueryType.GENERAL, 0.0
        
        query_lower = query.lower().strip()
        
        # Score each type by keyword matches
        scores: Dict[QueryType, int] = {}
        
        for qtype, keywords in self.TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            scores[qtype] = score
        
        # Find the best match
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        
        if best_score == 0:
            return QueryType.GENERAL, 0.5  # Default with neutral confidence
        
        # Confidence = matches / keywords checked (capped at 1.0)
        total_keywords = len(self.TYPE_KEYWORDS.get(best_type, []))
        confidence = min(best_score / max(total_keywords * 0.15, 1), 1.0)
        
        return best_type, round(confidence, 3)
