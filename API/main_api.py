import sys, os
from dotenv import load_dotenv

# Ensure project root is importable when running: python main_api.py
API_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(API_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

load_dotenv()

from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
import json

from typing import List, Optional, Annotated

from models.response import SearchResponse
from models.entry_models import SearchEntryRequest, StageModalities


# Routers & shared models (already defined in routes/search_routes.py)
from routes.submit_csv_routes import router as submit_csv_router
from routes.search_routes import router as search_router
from routes.es_routes import router as es_router
from routes.llm_routes import router as llm_router


from retrieve_vitH import index as search_index, metadata as search_metadata

from results_utils import discard_duplicate_frame, events_chain, update_temporal_score
from utils import normalize_score, sort_score_results, get_weighted_union_results

from results_utils import process_one_stage


app = FastAPI(title="Text-to-Image Retrieval API", version="1.0.0")

# Mount existing routers
app.include_router(search_router)
app.include_router(submit_csv_router)
app.include_router(es_router)
app.include_router(llm_router)

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
async def search_entry(entry: Annotated[SearchEntryRequest, Form()], request: Request):
    try:
        collected_results = []

        # Access raw form for files while using parsed model for fields
        form = await request.form()
        
        stage_items = sorted(entry.stage_list.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0])
        # [(0, stage0), (1,stage1), (2,stage2), ...]
        
        for stage_key, modalities in stage_items:
            print("\nProcessing stage: ", stage_key)
            stage_result = await process_one_stage(modalities, form, entry.top_k)
            print("FINISH Process one stage")
            if stage_result is not None:
                stage_result = normalize_score(stage_result)
                collected_results.append(stage_result)
                
        flat_results = None
        print("HERE before event chaining")
        # apply event-chaining for multi-stage
        if len(stage_items) > 1:
            weight_list = [0.6] * len(stage_items) # keep top highest 60% quantity
            weighted_res_quant = get_weighted_union_results(collected_results, weight_list, fuse=False)
            event_seqs = events_chain(weighted_res_quant)
            print("EVENT CHAINED DONE !")
            flat_results = update_temporal_score(event_seqs)
                
        else: # single stage
            print("SingleStage only")
            flat_results = collected_results[0] if collected_results else []

        flat_results = discard_duplicate_frame(flat_results)
        flat_results = normalize_score(flat_results)
        sorted_results = sort_score_results(flat_results, reverse=True)
        
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
