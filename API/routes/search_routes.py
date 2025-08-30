from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

# Parent directory to import retrieve module, make script-friendly
API_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(API_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from retrieve_vitL import search_query, index, metadata, ImageResult


router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 100


class SearchResponse(BaseModel):
    success: bool
    query: str
    results: List[ImageResult]
    total_results: int
    message: Optional[str] = None


@router.post("/text", response_model=SearchResponse)
async def text_search(request: SearchRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty")

    try:
        if index is None or metadata is None:
            raise HTTPException(
                status_code=500,
                detail="Search index not loaded. Please ensure indexing has been completed."
            )

        en_query = request.query.strip()
        results = search_query(en_query.strip().lower(), index, metadata, top_k=request.top_k)

        return SearchResponse(
            success=True,
            query=request.query.strip(),
            results=results,
            total_results=len(results),
            message=f"Found {len(results)} results for query: '{request.query.strip()}'"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error during search: {str(e)}")


