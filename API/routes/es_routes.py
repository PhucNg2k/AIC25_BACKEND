from models.request import *
from models.response import *

from fastapi import  HTTPException, APIRouter

import sys
import os

# Parent directory to import retrieve module, make script-friendly
API_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(API_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from dependecies import OCRClientDeps, ASRClientDeps
from utils import convert_ImageList
from dotenv import load_dotenv
load_dotenv()



router = APIRouter(prefix="/es-search", tags=["elastic search"])

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

@router.post("/video_name", response_model=SearchResponse)
async def search_video_name(request: SearchRequest, es_client: OCRClientDeps):
    try:
        query_text = request.value
        top_k = request.top_k
        
        search_body = make_videoname_search_body(query_text, top_k)

        raw_results = es_client.search_parsed(search_body)

        results = convert_ImageList(raw_results)

        return SearchResponse(
            success=True,
            query=request.value.strip(),
            results=results,
            total_results=len(results),
            message=f"Found {len(results)} results for query: '{request.value.strip()}'"
        )
    except Exception as e:
        # Log the actual error for debugging
        print(f"Error in search_ocr: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

        

@router.post("/ocr", response_model=SearchResponse)
async def search_ocr(request: SearchRequest, es_client: OCRClientDeps):
    try:
        query_text = request.value
        top_k = request.top_k
        
        search_body = make_ocr_search_body(query_text, top_k)

        raw_results = es_client.search_parsed(search_body)

        results = convert_ImageList(raw_results)

        return SearchResponse(
            success=True,
            query=request.value.strip(),
            results=results,
            total_results=len(results),
            message=f"Found {len(results)} results for query: '{request.value.strip()}'"
        )
    except Exception as e:
        # Log the actual error for debugging
        print(f"Error in search_ocr: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

