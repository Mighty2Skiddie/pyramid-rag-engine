"""
Bonus — Reasoning Adapter (Main Router)

The core routing layer that sits between user queries and domain-specific
reasoning handlers. Implements the Strategy Pattern for AI reasoning:
    1. Classify the query to detect its domain
    2. Route to the appropriate handler
    3. Optionally chain handlers for multi-step reasoning
    4. Log all routing decisions for monitoring

Architecture Principles:
    - Plug-and-play: New handlers are registered in HANDLER_REGISTRY
      with zero changes to existing code
    - Swappable classifier: Start with keyword rules, upgrade to ML later
    - Chain-aware: Handlers can receive context from previous handlers
    - Observable: All routing decisions are logged with confidence scores
"""

from typing import Dict, Optional, List

from bonus_reasoning_adapter.interfaces import (
    QueryType,
    ReasoningHandler,
    ReasoningResponse
)
from bonus_reasoning_adapter.classifier import KeywordQueryClassifier
from bonus_reasoning_adapter.handlers import (
    MathReasoningHandler,
    LegalReasoningHandler,
    FactualReasoningHandler,
    CodeReasoningHandler,
    DefaultReasoningHandler
)
from shared.logger import get_logger

logger = get_logger(__name__)


class ReasoningAdapter:
    """
    Meta-routing layer for domain-aware query processing.
    
    The adapter:
        1. Classifies incoming queries by domain type
        2. Dispatches to the registered handler for that domain
        3. Supports handler chaining (e.g., FACTUAL → MATH)
        4. Falls back to the default handler if no match is found
    
    Adding a new domain:
        1. Create a new handler class implementing ReasoningHandler
        2. Register it: adapter.register_handler(MyHandler())
        3. Add keywords to the classifier (if using keyword-based)
    
    Example:
        >>> adapter = ReasoningAdapter()
        >>> response = adapter.route("What is 25 + 37?")
        >>> print(response.handler_name)  # "MathReasoningHandler"
        >>> print(response.answer)        # "The sum is 62.0"
    """
    
    def __init__(self):
        """Initialize with default handlers and keyword classifier."""
        self.classifier = KeywordQueryClassifier()
        self.routing_log: List[Dict] = []
        
        # Handler registry — maps QueryType → handler instance
        self._registry: Dict[QueryType, ReasoningHandler] = {}
        
        # Register all built-in handlers
        self.register_handler(MathReasoningHandler())
        self.register_handler(LegalReasoningHandler())
        self.register_handler(FactualReasoningHandler())
        self.register_handler(CodeReasoningHandler())
        self.register_handler(DefaultReasoningHandler())
        
        logger.info(
            f"ReasoningAdapter initialized with {len(self._registry)} handlers: "
            f"{[h.name for h in self._registry.values()]}"
        )
    
    def register_handler(self, handler: ReasoningHandler) -> None:
        """
        Register a new domain handler. Replaces existing handler
        for the same QueryType if one exists.
        
        Args:
            handler: ReasoningHandler instance to register
        """
        self._registry[handler.supported_type] = handler
        logger.info(f"Registered handler: {handler.name} for {handler.supported_type.value}")
    
    def route(self, query: str, context: Optional[str] = None) -> ReasoningResponse:
        """
        Route a query to the appropriate reasoning handler.
        
        Process:
            1. Classify the query type
            2. Look up the registered handler
            3. Invoke the handler with the query (and optional context)
            4. Log the routing decision
            5. Return the response
        
        Args:
            query: User's input query
            context: Optional context from a prior step (e.g., retrieved docs)
        
        Returns:
            ReasoningResponse from the selected handler
        """
        # Step 1: Classify
        query_type, confidence = self.classifier.classify(query)
        
        logger.info(
            f"Query classified as {query_type.value} "
            f"(confidence: {confidence:.3f})"
        )
        
        # Step 2: Look up handler (fall back to GENERAL if not found)
        handler = self._registry.get(
            query_type,
            self._registry.get(QueryType.GENERAL)
        )
        
        # Step 3: Invoke handler
        response = handler.handle(query, context)
        
        # Step 4: Log routing decision
        log_entry = {
            "query": query[:100],
            "detected_type": query_type.value,
            "classifier_confidence": confidence,
            "handler": handler.name,
            "response_confidence": response.confidence
        }
        self.routing_log.append(log_entry)
        
        logger.info(
            f"Routed to {handler.name} → confidence: {response.confidence:.3f}"
        )
        
        return response
    
    def route_with_chain(
        self,
        query: str,
        chain: List[QueryType]
    ) -> ReasoningResponse:
        """
        Route a query through a chain of handlers sequentially.
        
        Each handler in the chain receives the previous handler's
        answer as context. This enables multi-step reasoning:
            e.g., FACTUAL → MATH: First retrieve relevant data,
            then perform calculations on it.
        
        Args:
            query: User's input query
            chain: Ordered list of QueryTypes to process through
        
        Returns:
            Final ReasoningResponse from the last handler in the chain
        """
        context = None
        response = None
        
        for step, qtype in enumerate(chain, 1):
            handler = self._registry.get(qtype)
            if handler is None:
                logger.warning(f"Chain step {step}: No handler for {qtype.value}")
                continue
            
            logger.info(f"Chain step {step}/{len(chain)}: {handler.name}")
            response = handler.handle(query, context)
            context = response.answer  # Pass answer as context to next handler
        
        return response
    
    def get_routing_stats(self) -> Dict:
        """
        Get statistics on routing decisions for monitoring.
        
        Returns:
            Dictionary with handler usage counts and average confidence
        """
        if not self.routing_log:
            return {"total_queries": 0}
        
        stats = {"total_queries": len(self.routing_log)}
        
        # Count by handler
        handler_counts = {}
        for entry in self.routing_log:
            h = entry["handler"]
            handler_counts[h] = handler_counts.get(h, 0) + 1
        
        stats["handler_usage"] = handler_counts
        stats["avg_confidence"] = round(
            sum(e["response_confidence"] for e in self.routing_log) / len(self.routing_log),
            3
        )
        
        return stats


# ────────────────────────────────────────────────────────────────
#  Demo / Standalone Usage
# ────────────────────────────────────────────────────────────────

def demo_adapter():
    """Run a demonstration of the reasoning adapter routing system."""
    
    print("\n" + "═" * 70)
    print("   BONUS — Reasoning Adapter Demo (Plug-and-Play Routing)")
    print("═" * 70)
    
    adapter = ReasoningAdapter()
    
    # Test queries spanning different domains
    test_queries = [
        "What is the sum of 145 and 287?",
        "What are the legal implications of contract breach?",
        "Who invented the telephone?",
        "How do I implement a binary search in Python?",
        "Tell me about the weather today",
        "Calculate the average of 10, 20, 30, 40, 50",
    ]
    
    for query in test_queries:
        response = adapter.route(query)
        
        print(f"\n{'─'*70}")
        print(f"  Query:   {query}")
        print(f"  Type:    {response.query_type.value}")
        print(f"  Handler: {response.handler_name}")
        print(f"  Answer:  {response.answer[:150]}")
        print(f"  Confidence: {response.confidence:.3f}")
        print(f"  Trace:   {' → '.join(response.reasoning_trace[:3])}")
    
    # Demo handler chaining
    print(f"\n{'═'*70}")
    print("  Handler Chaining Demo: FACTUAL → MATH")
    print(f"{'═'*70}")
    
    chain_query = "How much is 500 plus the population of Iceland?"
    chain_response = adapter.route_with_chain(
        chain_query,
        [QueryType.FACTUAL, QueryType.MATH]
    )
    print(f"  Query: {chain_query}")
    print(f"  Final Handler: {chain_response.handler_name}")
    print(f"  Answer: {chain_response.answer[:200]}")
    
    # Routing stats
    print(f"\n{'─'*70}")
    stats = adapter.get_routing_stats()
    print(f"  Routing Stats: {stats}")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    demo_adapter()
