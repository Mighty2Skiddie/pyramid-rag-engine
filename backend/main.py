"""
Vexoo AI Platform — FastAPI Backend

Serves the Part 1 document pipeline and bonus reasoning adapter
as REST API endpoints for the Next.js frontend.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import pyramid, reasoning

app = FastAPI(
    title="Vexoo AI Platform API",
    description="Document Intelligence + Math Reasoning — REST API",
    version="0.1.0",
)

# CORS — allow the Next.js dev server and production domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(pyramid.router, prefix="/api/pyramid", tags=["Document Pipeline"])
app.include_router(reasoning.router, prefix="/api/reasoning", tags=["Reasoning Adapter"])


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "vexoo-ai-platform"}
