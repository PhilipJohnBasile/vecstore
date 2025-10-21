"""
FastAPI Integration Example

This example demonstrates building a REST API for vector search
using FastAPI and VecStore.

Install dependencies:
    pip install fastapi uvicorn

Run the server:
    python fastapi_app.py

Test with curl:
    curl -X POST http://localhost:8000/upsert \\
      -H "Content-Type: application/json" \\
      -d '{"id": "doc1", "vector": [0.1, 0.2, 0.3], "metadata": {"text": "Hello"}}'

    curl -X POST http://localhost:8000/query \\
      -H "Content-Type: application/json" \\
      -d '{"vector": [0.1, 0.2, 0.3], "k": 5}'
"""

import random
from typing import Any, Dict, List
from contextlib import asynccontextmanager

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Error: FastAPI not installed")
    print("Install with: pip install fastapi uvicorn")
    exit(1)

from vecstore import VecStore

# Global store instance
store: VecStore = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app"""
    global store
    print("Starting up... Creating VecStore")
    store = VecStore("./fastapi_vecstore_db")
    yield
    print("Shutting down...")


app = FastAPI(
    title="VecStore API",
    description="REST API for vector similarity search",
    version="0.0.1",
    lifespan=lifespan,
)


# Request/Response models
class UpsertRequest(BaseModel):
    id: str
    vector: List[float]
    metadata: Dict[str, Any]


class QueryRequest(BaseModel):
    vector: List[float]
    k: int = 10
    filter: str | None = None


class SearchResultResponse(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    results: List[SearchResultResponse]
    count: int


# Endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "VecStore API is running",
        "version": "0.0.1",
    }


@app.post("/upsert")
async def upsert(request: UpsertRequest):
    """
    Insert or update a vector.

    Example:
        POST /upsert
        {
            "id": "doc1",
            "vector": [0.1, 0.2, 0.3],
            "metadata": {"text": "Hello world"}
        }
    """
    try:
        store.upsert(request.id, request.vector, request.metadata)
        return {
            "status": "success",
            "id": request.id,
            "message": f"Vector '{request.id}' upserted successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Search for similar vectors.

    Example:
        POST /query
        {
            "vector": [0.1, 0.2, 0.3],
            "k": 5,
            "filter": "category = 'tech'"
        }
    """
    try:
        results = store.query(
            vector=request.vector,
            k=request.k,
            filter=request.filter,
        )

        response_results = [
            SearchResultResponse(
                id=r.id,
                score=r.score,
                metadata=r.metadata,
            )
            for r in results
        ]

        return QueryResponse(
            results=response_results,
            count=len(response_results),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/vector/{vector_id}")
async def delete_vector(vector_id: str):
    """Delete a vector by ID"""
    try:
        store.remove(vector_id)
        return {
            "status": "success",
            "message": f"Vector '{vector_id}' deleted",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def stats():
    """Get store statistics"""
    return {
        "vector_count": len(store),
        "is_empty": store.is_empty(),
    }


@app.post("/optimize")
async def optimize():
    """Optimize the store by removing deleted entries"""
    try:
        removed_count = store.optimize()
        return {
            "status": "success",
            "removed_count": removed_count,
            "message": f"Optimized store, removed {removed_count} entries",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the FastAPI server"""
    print("=" * 60)
    print("Starting VecStore FastAPI Server")
    print("=" * 60)
    print("\nAPI Documentation:")
    print("  - Interactive docs: http://localhost:8000/docs")
    print("  - OpenAPI schema: http://localhost:8000/openapi.json")
    print("\nPress CTRL+C to stop\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )


if __name__ == "__main__":
    main()
