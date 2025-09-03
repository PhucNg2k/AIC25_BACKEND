import sys, os

# Ensure project root is importable when running: python search_api.py
API_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(API_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import httpx

from typing import List, Optional, Annotated
from models import *

# Routers & shared models (already defined in routes/search_routes.py)
from routes.submit_csv_routes import router as submit_csv_router
from routes.search_routes import router as search_router
from routes.es_routes import router as es_router

from retrieve_vitL import index as search_index, metadata as search_metadata

from utils import get_intersection_results, sort_descending_results
from itertools import chain

app = FastAPI(title="Text-to-Image Retrieval API", version="1.0.0")

# Mount existing routers
app.include_router(search_router)
app.include_router(submit_csv_router)
app.include_router(es_router)

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
async def search_entry(
    text: Annotated[Optional[str], Form()] = None,
    img: Annotated[Optional[UploadFile], File()] = None,
    ocr: Annotated[Optional[str], Form()] = None,
    localized: Annotated[Optional[str], Form()] = None,
    top_k: Annotated[int, Form()] = 100,
    intersect: Annotated[bool, Form()] = False,
    order_dict: Annotated[dict, Form()] = None,
    weight_dict: Annotated[dict, Form()] = None
):
    if not any([text, ocr, localized, img]):
        raise HTTPException(
            status_code=400,
            detail="At least one search modality (text, ocr, localized, or img) must be provided",
        )
    
    try:
        results: List[ImageResult] = []

        # Text modality - HTTP call to /search/text endpoint
        if text and text.strip():
            async with httpx.AsyncClient() as client:
                text_response = await client.post(
                    "http://localhost:8000/search/text",
                    json={"query": text.strip(), "top_k": top_k}
                )
                if text_response.status_code == 200:
                    text_data = text_response.json()

                    if (text_data.get("results", [])!=[]):
                        results.append(text_data["results"])
                else:
                    raise HTTPException(
                        status_code=text_response.status_code,
                        detail=f"Text search failed: {text_response.text}"
                    )

        # TODO: Implement these modalities later
        if ocr and ocr.strip():
            async with httpx.AsyncClient() as client:
                text_response = await client.post(
                    "http://localhost:8000/es-search/ocr",
                    json={"query": ocr.strip(), "top_k": top_k}
                )
                if text_response.status_code == 200:
                    text_data = text_response.json()

                    if (text_data.get("results", [])!=[]):
                        results.append(text_data["results"])
                else:
                    raise HTTPException(
                        status_code=text_response.status_code,
                        detail=f"Text search failed: {text_response.text}"
                        )

        if localized and localized.strip():
            pass
        
        if img:
            async with httpx.AsyncClient() as client:
                files = {
                    "image_file": (
                        img.filename or "upload",
                        img.file,
                        getattr(img, "content_type", "application/octet-stream"),
                    )
                }
                image_response = await client.post(
                    "http://localhost:8000/search/image",
                    files=files,
                    data={"top_k": str(top_k)},
                )
                if image_response.status_code == 200:
                    image_data = image_response.json()

                    if (image_data.get("results", []) != []):
                        results.append(image_data["results"])
                else:
                    raise HTTPException(
                        status_code=image_response.status_code,
                        detail=f"Image search failed: {image_response.text}"
                    )
        
        if (intersect):
            results = get_intersection_results(results)
        else:
            results = list(chain.from_iterable(results)) # flatten out the results
        
        results = sort_descending_results(results)

        success_status = (len(results) > 0) 
        
        return SearchResponse(
            success=success_status,
            query="multi-modal search",
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
