import sys, os

# Ensure project root is importable when running: python search_api.py
API_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(API_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import httpx

from typing import List
from models import *

# Routers & shared models (already defined in routes/search_routes.py)
from routes.submit_csv_routes import router as submit_csv_router
from routes.search_routes import router as search_router

from retrieve_vitL import index as search_index, metadata as search_metadata

app = FastAPI(title="Text-to-Image Retrieval API", version="1.0.0")

# Mount existing routers
app.include_router(search_router)
app.include_router(submit_csv_router)

# CORS: if you need cookies/Authorization headers, replace ["*"] with your exact origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # "*" + credentials=True is blocked by browsers
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Text-to-Image Retrieval API is running", "status": "healthy"}

@app.post("/search-entry", response_model=SearchResponse)
async def search_entry(request: SearchRequestEntry):
    try:
        results: List[ImageResult] = []

        # Text modality - HTTP call to /search/text endpoint
        if request.text and request.text.strip():
            async with httpx.AsyncClient() as client:
                text_response = await client.post(
                    "http://localhost:8000/search/text",
                    json={"query": request.text.strip(), "top_k": request.top_k}
                )
                if text_response.status_code == 200:
                    text_data = text_response.json()
                    
                    results.extend(text_data["results"])
                else:
                    raise HTTPException(
                        status_code=text_response.status_code,
                        detail=f"Text search failed: {text_response.text}"
                    )

        # TODO: Implement these modalities later
        if request.ocr and request.ocr.strip():
            pass

        if request.localized and request.localized.strip():
            pass
        
        if request.img:
            pass

        if not any([request.text, request.ocr, request.localized, request.img]):
            raise HTTPException(
                status_code=400,
                detail="At least one search modality (text, ocr, localized, or img) must be provided",
            )

        return SearchResponse(
            success=True,
            query=request.text or request.ocr or request.localized or "multi-modal search",
            results=results,
            total_results=len(results),
            message=f"Found {len(results)} results from search gateway",
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during /search: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during /search: {str(e)}")

@app.post("/reranker")
async def rerank(req: RerankRequest):
    # TODO: wire this to your reranker
    return {"success": True, "count": min(len(req.chosen_frames), req.top_k)}



@app.get("/health")
async def health_check():
    try:
        index_status = "loaded" if search_index is not None else "not loaded"
        metadata_status = "loaded" if search_metadata is not None else "not loaded"
        total_images = len(search_metadata.keys())
        status = "healthy" if (search_index and search_metadata and total_images > 0) else 'unhealthy'

        return {
            "status": status,
            "index_status": index_status,
            "metadata_status": metadata_status,
            "total_images": total_images
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("Starting Text-to-Image Retrieval API...")
    print("API Documentation: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
