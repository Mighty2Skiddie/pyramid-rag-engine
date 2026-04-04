"""
Reasoning API router — Query routing via the bonus reasoning adapter.

Wraps the existing bonus_reasoning_adapter as a REST endpoint.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional

from bonus_reasoning_adapter.adapter import ReasoningAdapter

router = APIRouter()

# Singleton adapter instance with all handlers pre-registered
_adapter = ReasoningAdapter()


# ── Request / Response Models ─────────────────────────────────────

class SolveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)

class SolveResponse(BaseModel):
    query: str
    query_type: str
    handler_name: str
    answer: str
    confidence: float
    reasoning_trace: List[str]
    metadata: dict

class StatsResponse(BaseModel):
    total_queries: int
    type_distribution: dict
    avg_confidence: float


# ── Endpoints ─────────────────────────────────────────────────────

@router.post("/solve", response_model=SolveResponse)
async def solve(req: SolveRequest):
    """
    Route a query through the reasoning adapter and return
    the domain-specific response with full trace.
    """
    response = _adapter.route(req.query)

    return SolveResponse(
        query=req.query,
        query_type=response.query_type.value,
        handler_name=response.handler_name,
        answer=response.answer,
        confidence=round(response.confidence, 4),
        reasoning_trace=response.reasoning_trace,
        metadata=response.metadata,
    )


@router.get("/stats", response_model=StatsResponse)
async def stats():
    """
    Return aggregated routing statistics.
    """
    s = _adapter.get_routing_stats()
    return StatsResponse(
        total_queries=s.get("total_queries", 0),
        type_distribution=s.get("type_distribution", {}),
        avg_confidence=round(s.get("avg_confidence", 0.0), 4),
    )


@router.get("/domains")
async def list_domains():
    """
    List all registered reasoning domains and their handlers.
    """
    domains = []
    for qtype, handler in _adapter._handlers.items():
        domains.append({
            "type": qtype.value,
            "handler": handler.__class__.__name__,
            "description": handler.__class__.__doc__ or "",
        })
    return {"domains": domains}
