from typing import Annotated
import io
from PIL import Image
import numpy as np

import math
from models import ImageResult, SearchResponse, SearchRequest

from fastapi import Depends, FastAPI, HTTPException, UploadFile, Body
from fastapi.middleware.cors import CORSMiddleware
import sys, os

# Ensure project root is importable when running: python search_api.py
API_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(API_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from dependecies import OCRClientDep, ASRClientDep
from config import DATA_SOURCE

from dotenv import load_dotenv
load_dotenv()



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,   # "*" + credentials=True is blocked by browsers
    allow_methods=["*"],
    allow_headers=["*"],
)

def convert_ImageList(raw_results):
    results = []
    for raw_result in raw_results:
        result = ImageResult(**raw_result)
        results.append(result)
    return results

def make_videoname_search_body(query_text, top_k):
    if not top_k or top_k <= 0:
        top_k=10000

    search_body = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "video_name": {
                                    "query": query_text,
                                    "operator": "or"
                                }
                            }
                        },
                        {
                            "wildcard": {
                                "video_name": f"*{query_text}*"
                            }
                        }
                    ]
                }
            },
            "size": top_k,
            "_source": ["video_folder", "video_name", "frame_id", "text"]
        }

    return search_body

def make_ocr_search_body(query_text, top_k = 50, fuzziness="AUTO"):
    search_body =  {
        "query": {
            "match": {
                "text": {
                    "query": query_text,
                    "fuzziness": fuzziness
                }
            }
        },
        "size": top_k,
        "_source": ["video_folder", "video_name", "frame_id", "text"]
    }
    return search_body

@app.post("/video_name", response_model=SearchResponse)
async def search_video_name(request: SearchRequest, es_client: OCRClientDep):
    try:
        query_text = request.query
        top_k = request.top_k
        
        search_body = make_videoname_search_body(query_text, top_k)

        raw_results = es_client.search_parsed(search_body)

        results = convert_ImageList(raw_results)

        return SearchResponse(
            success=True,
            query=request.query.strip(),
            results=results,
            total_results=len(results),
            message=f"Found {len(results)} results for query: '{request.query.strip()}'"
        )
    except Exception as e:
        # Log the actual error for debugging
        print(f"Error in search_ocr: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

        

@app.post("/ocr", response_model=SearchResponse)
async def search_ocr(request: SearchRequest, es_client: OCRClientDep):
    try:
        query_text = request.query
        top_k = request.top_k
        
        search_body = make_ocr_search_body(query_text, top_k)

        raw_results = es_client.search_parsed(search_body)

        results = convert_ImageList(raw_results)

        return SearchResponse(
            success=True,
            query=request.query.strip(),
            results=results,
            total_results=len(results),
            message=f"Found {len(results)} results for query: '{request.query.strip()}'"
        )
    except Exception as e:
        # Log the actual error for debugging
        print(f"Error in search_ocr: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

        


if __name__ == "__main__":
    import uvicorn
    print("Starting Text-to-Image Retrieval API...")
    print("API Documentation: http://localhost:8001/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)