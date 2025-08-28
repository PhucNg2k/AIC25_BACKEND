from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import sys
import os
import csv
import json

# Ensure project root is importable when running as script: python search_api.py
API_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(API_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from routes.submit_csv_routes import router as submit_csv_router
from routes.search_routes import router as search_router
from routes.search_routes import text_search as text_search_route
from routes.search_routes import SearchRequest as RouterSearchRequest



app = FastAPI(title="Text-to-Image Retrieval API", version="1.0.0")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Text-to-Image Retrieval API is running", "status": "healthy"}

class SearchRequestEntry(BaseModel):
    text: Optional[str]=None
    img: Optional[str]=None
    ocr: Optional[str]=None
    localized: Optional[str]=None
    
    top_k: int = 80


@app.post("/search")
async def search_entry(request: SearchRequestEntry):
    if request.top_k <= 0 or request.top_k > 100:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 100")
    
    try:
        results = []
        
        # Handle text search modality
        if request.text and request.text.strip():
            print(f"Processing text search: '{request.text}'")
            
            # Create request for the router's text search
            text_search_request = RouterSearchRequest(query=request.text.strip(), top_k=request.top_k)
            # Call the router's text search function directly
            text_response = await text_search_route(text_search_request)
            results.extend(text_response.results)
            
        # Handle OCR search modality (placeholder for future implementation)
        if request.ocr and request.ocr.strip():
            print(f"OCR search requested: '{request.ocr}' (not implemented yet)")
            # TODO: Implement OCR search
            pass

        if request.localized and request.localized.strip():
            print(f"OCR search requested: '{request.localized}' (not implemented yet)")
            # TODO: Implement localized search
            pass
            
        # Handle image search modality (placeholder for future implementation)  
        if request.img:
            print(f"Image search requested (not implemented yet)")
            # TODO: Implement image-based search
            pass
        
        # If no valid search modalities provided
        if not any([request.text, request.ocr, request.img]):
            raise HTTPException(
                status_code=400, 
                detail="At least one search modality (text, ocr, or img) must be provided"
            )
        
        # For now, just return text search results
        # Later you can add aggregation/reranking logic here
        return SearchResponse(
            success=True,
            query=request.text or "multi-modal search",
            results=results,
            total_results=len(results),
            message=f"Found {len(results)} results from search gateway"
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        print(f"Error during search Entry: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during search Entry: {str(e)}"
        )

@app.post("/reranker")
async def rerank(chose_frames):
    pass


app.include_router(search_router)
app.include_router(submit_csv_router)

@app.get("/health")
async def health_check():
    """Extended health check with system status"""
    try:
        index_status = "loaded" if index is not None else "not loaded"
        metadata_status = "loaded" if metadata is not None else "not loaded"
        
        return {
            "status": "healthy",
            "index_status": index_status,
            "metadata_status": metadata_status,
            "total_images": metadata.get('num_images', 0) if metadata else 0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

    

if __name__ == "__main__":
    import uvicorn
    print("Starting Text-to-Image Retrieval API...")
    print("API Documentation available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)