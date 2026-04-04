"""
Bonus — Domain-Specific Reasoning Handlers

Each handler implements the ReasoningHandler interface and encapsulates
domain-specific reasoning logic. Handlers are designed to be:
    - Self-contained: Each handles one domain independently
    - Chainable: A handler can receive context from a previous handler
    - Swappable: Replace keyword logic with LLM calls without interface changes

Current Implementation: Simulated reasoning using templates and rule-based logic
Production Upgrade:     Each handler would invoke a domain-tuned LLM or API
"""

import re
from typing import Optional

from bonus_reasoning_adapter.interfaces import (
    ReasoningHandler,
    ReasoningResponse,
    QueryType
)


class MathReasoningHandler(ReasoningHandler):
    """
    Handles mathematical queries with step-by-step chain-of-thought.
    
    Demonstrates how a math-specific handler would break down problems
    into steps, apply arithmetic, and validate the numeric output.
    
    Production: Would use a fine-tuned math model or Wolfram Alpha API.
    """
    
    @property
    def name(self) -> str:
        return "MathReasoningHandler"
    
    @property
    def supported_type(self) -> QueryType:
        return QueryType.MATH
    
    def handle(self, query: str, context: Optional[str] = None) -> ReasoningResponse:
        """Process a math query with chain-of-thought reasoning."""
        
        trace = [
            f"Step 1: Identified query as MATH type",
            f"Step 2: Parsing mathematical components from query",
        ]
        
        # Attempt to extract and solve simple arithmetic
        numbers = re.findall(r'\d+\.?\d*', query)
        
        if numbers:
            nums = [float(n) for n in numbers]
            trace.append(f"Step 3: Extracted numbers: {nums}")
            
            # Detect operation from keywords
            q_lower = query.lower()
            if any(w in q_lower for w in ["sum", "add", "total", "plus"]):
                result = sum(nums)
                trace.append(f"Step 4: Operation = SUM → {' + '.join(numbers)} = {result}")
                answer = f"The sum is {result}"
            elif any(w in q_lower for w in ["multiply", "product", "times"]):
                result = 1
                for n in nums:
                    result *= n
                trace.append(f"Step 4: Operation = MULTIPLY → result = {result}")
                answer = f"The product is {result}"
            elif any(w in q_lower for w in ["subtract", "minus", "difference"]):
                result = nums[0] - sum(nums[1:]) if len(nums) > 1 else nums[0]
                trace.append(f"Step 4: Operation = SUBTRACT → result = {result}")
                answer = f"The difference is {result}"
            elif any(w in q_lower for w in ["average", "mean"]):
                result = sum(nums) / len(nums)
                trace.append(f"Step 4: Operation = AVERAGE → {sum(nums)}/{len(nums)} = {result}")
                answer = f"The average is {result}"
            else:
                trace.append(f"Step 4: No clear operation detected, presenting numbers")
                answer = f"Extracted values: {nums}. Please specify the operation."
            
            confidence = 0.85
        else:
            trace.append("Step 3: No numeric values found in query")
            answer = ("This appears to be a math question, but I couldn't extract "
                     "numeric values. In production, this would be sent to a math "
                     "reasoning model (e.g., fine-tuned on GSM8K).")
            confidence = 0.3
        
        trace.append(f"Step 5: Confidence = {confidence}")
        
        return ReasoningResponse(
            answer=answer,
            reasoning_trace=trace,
            confidence=confidence,
            handler_name=self.name,
            query_type=QueryType.MATH
        )


class LegalReasoningHandler(ReasoningHandler):
    """
    Handles legal queries with clause-sensitive retrieval and disclaimers.
    
    Demonstrates domain-specific prompt templating, source attribution,
    and mandatory legal disclaimers.
    
    Production: Would integrate with a legal knowledge base and add
    jurisdiction-aware filtering.
    """
    
    @property
    def name(self) -> str:
        return "LegalReasoningHandler"
    
    @property
    def supported_type(self) -> QueryType:
        return QueryType.LEGAL
    
    def handle(self, query: str, context: Optional[str] = None) -> ReasoningResponse:
        """Process a legal query with domain-specific reasoning."""
        
        trace = [
            "Step 1: Identified query as LEGAL type",
            "Step 2: Activating clause-sensitive retrieval template",
            "Step 3: Filtering pyramid nodes by L3 category = 'legal'",
        ]
        
        if context:
            trace.append(f"Step 4: Using provided context from previous handler")
            answer = (
                f"Based on the retrieved legal content: {context[:200]}... "
                f"\n\nNote: This response is based on document retrieval. "
                f"For legal advice, consult a qualified attorney."
            )
            confidence = 0.7
        else:
            trace.append("Step 4: No context provided — generating template response")
            answer = (
                f"Your legal query: '{query}'\n\n"
                f"In a production system, this handler would:\n"
                f"1. Filter the Knowledge Pyramid for legal-category nodes (L3)\n"
                f"2. Retrieve relevant clauses and regulations\n"
                f"3. Present findings with source citations\n\n"
                f"⚠️ DISCLAIMER: This system does not provide legal advice. "
                f"Consult a qualified attorney for legal matters."
            )
            confidence = 0.4
        
        return ReasoningResponse(
            answer=answer,
            reasoning_trace=trace,
            confidence=confidence,
            handler_name=self.name,
            query_type=QueryType.LEGAL,
            metadata={"disclaimer_added": True}
        )


class FactualReasoningHandler(ReasoningHandler):
    """
    Handles factual queries with direct pyramid L1/L2 retrieval.
    
    Focuses on precision — returns answers only when confidence
    exceeds a threshold, otherwise reports insufficient evidence.
    """
    
    @property
    def name(self) -> str:
        return "FactualReasoningHandler"
    
    @property
    def supported_type(self) -> QueryType:
        return QueryType.FACTUAL
    
    def handle(self, query: str, context: Optional[str] = None) -> ReasoningResponse:
        """Process a factual query with direct retrieval."""
        
        trace = [
            "Step 1: Identified query as FACTUAL type",
            "Step 2: Performing direct retrieval from pyramid L1 (raw text) and L2 (summaries)",
        ]
        
        if context:
            trace.append("Step 3: Context available — extracting factual answer")
            answer = f"Based on available information: {context[:300]}"
            confidence = 0.75
        else:
            trace.append("Step 3: No context provided — in production, would query pyramid")
            answer = (
                f"Query: '{query}'\n\n"
                f"This handler retrieves factual information directly from the "
                f"Knowledge Pyramid's L1 (raw text) and L2 (summary) layers, "
                f"prioritizing precision over recall."
            )
            confidence = 0.3
        
        trace.append(f"Step 4: Confidence check — {confidence:.2f} "
                     f"{'(sufficient)' if confidence >= 0.5 else '(below threshold)'}")
        
        return ReasoningResponse(
            answer=answer,
            reasoning_trace=trace,
            confidence=confidence,
            handler_name=self.name,
            query_type=QueryType.FACTUAL
        )


class CodeReasoningHandler(ReasoningHandler):
    """
    Handles code-related queries with structured technical responses.
    """
    
    @property
    def name(self) -> str:
        return "CodeReasoningHandler"
    
    @property
    def supported_type(self) -> QueryType:
        return QueryType.CODE
    
    def handle(self, query: str, context: Optional[str] = None) -> ReasoningResponse:
        trace = [
            "Step 1: Identified query as CODE type",
            "Step 2: Activating technical reasoning template",
            "Step 3: Would analyze code context, detect language, and provide solution"
        ]
        
        answer = (
            f"Code query detected: '{query}'\n\n"
            f"In production, this handler would:\n"
            f"1. Detect the programming language\n"
            f"2. Parse the technical context\n"
            f"3. Generate or review code with explanations\n"
            f"4. Validate syntax and logic"
        )
        
        return ReasoningResponse(
            answer=answer,
            reasoning_trace=trace,
            confidence=0.4,
            handler_name=self.name,
            query_type=QueryType.CODE
        )


class DefaultReasoningHandler(ReasoningHandler):
    """
    Fallback handler for general queries.
    Uses standard RAG retrieval without domain-specific logic.
    """
    
    @property
    def name(self) -> str:
        return "DefaultHandler"
    
    @property
    def supported_type(self) -> QueryType:
        return QueryType.GENERAL
    
    def handle(self, query: str, context: Optional[str] = None) -> ReasoningResponse:
        trace = [
            "Step 1: No specific domain detected — using default handler",
            "Step 2: Standard RAG retrieval across all pyramid levels"
        ]
        
        answer = (
            f"Query: '{query}'\n\n"
            f"Processed via standard RAG retrieval. In production, this would "
            f"search the full Knowledge Pyramid without domain-specific filtering."
        )
        
        return ReasoningResponse(
            answer=answer,
            reasoning_trace=trace,
            confidence=0.5,
            handler_name=self.name,
            query_type=QueryType.GENERAL
        )
