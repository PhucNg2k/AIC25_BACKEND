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

from retrieve_vitL import search_query, index, metadata
from models import *

router = APIRouter(prefix="/search", tags=["search"])

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
        raw_results = search_query(en_query.strip().lower(), index, metadata, top_k=request.top_k)
        
        # Convert raw results to ImageResult instances
        results = []
        for raw_result in raw_results:
            result = ImageResult(**raw_result)
            results.append(result)

        return SearchResponse(
            success=True,
            query=request.query.strip(),
            results=results,
            total_results=len(results),
            message=f"Found {len(results)} results for query: '{request.query.strip()}'"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error during search: {str(e)}")


