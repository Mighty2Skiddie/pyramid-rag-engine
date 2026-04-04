"""
Pyramid API router — Document ingestion, querying, and exploration.

Wraps the existing Part 1 Python modules as REST endpoints.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Optional, List

from part1_document_pipeline.input_layer import load_document
from part1_document_pipeline.chunker import chunk_document
from part1_document_pipeline.pyramid_builder import build_pyramid
from part1_document_pipeline.retriever import query_pyramid
from shared.config_manager import load_config

from backend.services.session_store import store

router = APIRouter()
config = load_config()


# ── Request / Response Models ─────────────────────────────────────

class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=10, max_length=50000)

class QueryRequest(BaseModel):
    session_id: str
    query: str = Field(..., min_length=1, max_length=1000)

class ChunkSummary(BaseModel):
    chunk_id: str
    raw_text: str
    summary: str
    category: str
    category_confidence: float
    keywords: List[str]

class IngestResponse(BaseModel):
    session_id: str
    chunk_count: int
    chunks: List[ChunkSummary]

class QueryResultItem(BaseModel):
    chunk_id: str
    score: float
    best_level: str
    level_scores: dict
    raw_text: str
    summary: str
    category: str
    keywords: List[str]

class QueryResponse(BaseModel):
    query: str
    results: List[QueryResultItem]
    result_count: int


# ── Endpoints ─────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(req: IngestTextRequest):
    """
    Accept raw text, build the Knowledge Pyramid, and return
    a session_id for subsequent queries.
    """
    try:
        doc = load_document(req.text)
        chunks = chunk_document(doc, config.chunker)
        pyramid_index = build_pyramid(chunks, config.pyramid)

        # Build chunk summaries for the response
        chunk_summaries = []
        for chunk in chunks:
            node = pyramid_index.get(chunk.chunk_id)
            if node:
                chunk_summaries.append(ChunkSummary(
                    chunk_id=chunk.chunk_id,
                    raw_text=node.raw_text[:500],  # Cap for JSON size
                    summary=node.summary,
                    category=node.category,
                    category_confidence=node.category_confidence,
                    keywords=node.keywords[:10],
                ))

        metadata = {
            "doc_id": doc.doc_id,
            "char_count": doc.char_count,
            "token_count": doc.token_count,
        }

        session_id = store.create(pyramid_index, chunks, metadata)

        return IngestResponse(
            session_id=session_id,
            chunk_count=len(chunks),
            chunks=chunk_summaries,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/ingest-file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """
    Accept a .txt or .pdf file upload and build the pyramid.
    """
    if file.size and file.size > 500_000:
        raise HTTPException(status_code=413, detail="File too large. Max 500KB.")

    content = await file.read()
    text = content.decode("utf-8", errors="replace")

    if len(text.strip()) < 10:
        raise HTTPException(status_code=400, detail="File is empty or too short.")

    # Reuse the text ingestion logic
    req = IngestTextRequest(text=text)
    return await ingest_document(req)


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Query an existing pyramid by session_id.
    """
    session = store.get(req.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired. Please re-upload the document.")

    results = query_pyramid(req.query, session.pyramid_index, config.retriever)

    items = []
    for r in results:
        items.append(QueryResultItem(
            chunk_id=r.chunk_id,
            score=round(r.score, 4),
            best_level=r.best_level,
            level_scores={k: round(v, 4) for k, v in r.level_scores.items()},
            raw_text=r.raw_text[:500],
            summary=r.summary,
            category=r.category,
            keywords=r.keywords[:10],
        ))

    return QueryResponse(
        query=req.query,
        results=items,
        result_count=len(items),
    )


@router.get("/explore/{session_id}/{level}")
async def explore_level(session_id: str, level: int):
    """
    Browse all chunks at a specific pyramid level (1–4).
    """
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired.")

    if level not in (1, 2, 3, 4):
        raise HTTPException(status_code=400, detail="Level must be 1, 2, 3, or 4.")

    level_map = {1: "raw_text", 2: "summary", 3: "category", 4: "keywords"}
    field = level_map[level]

    data = []
    for node in session.pyramid_index.values():
        entry = {"chunk_id": node.chunk_id}
        if level == 1:
            entry["content"] = node.raw_text[:500]
        elif level == 2:
            entry["content"] = node.summary
        elif level == 3:
            entry["content"] = node.category
            entry["confidence"] = round(node.category_confidence, 3)
        elif level == 4:
            entry["keywords"] = node.keywords[:10]
            entry["embedding_dim"] = len(node.embedding) if node.embedding is not None else 0
        data.append(entry)

    return {"session_id": session_id, "level": level, "level_name": f"L{level}", "chunks": data}
