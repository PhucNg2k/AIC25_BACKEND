import sys, os

# Ensure project root is importable when running: python search_api.py
API_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(API_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
import json

from typing import List, Optional, Annotated

from models.response import SearchResponse
from models.entry_models import SearchEntryRequest, StageModalities
from search_utils import call_text_search, call_ocr_search, call_asr_search, call_image_upload

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
async def search_entry(request: Request):
    try:
        collected_results = []
        # If multipart form-data is sent (UI path): parse stage_list JSON and per-stage image files
        if request.headers.get("content-type", "").startswith("multipart/form-data"):
            form = await request.form()
            top_k = int(form.get("top_k", 30))
            stage_list_str = form.get("stage_list")
            if stage_list_str:
                try:
                    stage_list_obj = json.loads(stage_list_str)
                except Exception:
                    raise HTTPException(status_code=400, detail="Invalid stage_list JSON in form-data")

                entry = SearchEntryRequest(stage_list=stage_list_obj, top_k=top_k)
                stage_items = sorted(entry.stage_list.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0])
                for stage_key, modalities in stage_items:
                    if not isinstance(modalities, StageModalities):
                        continue

                    if modalities.text and modalities.text.value:
                        text_results = await call_text_search(modalities.text.value, entry.top_k)
                        if text_results:
                            collected_results.append(text_results)

                    if modalities.ocr and modalities.ocr.value:
                        ocr_results = await call_ocr_search(modalities.ocr.value, entry.top_k)
                        if ocr_results:
                            collected_results.append(ocr_results)

                    if modalities.asr and modalities.asr.value:
                        asr_results = await call_asr_search(modalities.asr.value, entry.top_k)
                        if asr_results:
                            collected_results.append(asr_results)

                    if modalities.img and modalities.img.value:
                        # UI sets img.value to the field name (e.g., img_1). Retrieve that file from form-data
                        field_name = modalities.img.value
                        img_file = form.get(field_name)
                        if img_file is not None:
                            img_results = await call_image_upload(img_file, entry.top_k)
                            if img_results:
                                collected_results.append(img_results)

        flat_results = list(chain.from_iterable(collected_results)) if collected_results else []
        sorted_results = sort_descending_results(flat_results)

        return SearchResponse(
            success=len(sorted_results) > 0,
            query="multi-modal search",
            results=sorted_results,
            total_results=len(sorted_results),
            message=f"Found {len(sorted_results)} results from search gateway",
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during /search-entry: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during /search-entry: {str(e)}")



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
