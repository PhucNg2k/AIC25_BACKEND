from models.request import *
from models.response import *

from fastapi import  HTTPException, APIRouter
import sys
import os
# NEW: bring in the aggregation search + frame utils
from search_frames_ocr import build_frame_agg_query
from frame_utils import get_metakey, get_pts_time, get_frame_path
import os

# Parent directory to import retrieve module, make script-friendly
API_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(API_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from dependecies import (
    OCRClientDeps,
    # ASRClientDeps
)
from utils import convert_ImageList
from group_utils import get_group_frames
# from load_embed_model import get_asr_embedding
# from asr_tools import get_estimate_keyframes

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



def make_asr_search_body(query_text, top_k=50, fuzziness="AUTO"):
    query_vector = get_asr_embedding(query_text)
    
    search_body = {
        "size": top_k,
        "_source": ["video_name", "text"],
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "text": {
                                "query": query_text,
                                "fuzziness": fuzziness,
                                "boost": 2.0   # lexical importance
                            }
                        }
                    },
                    {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                "params": {"query_vector": query_vector}
                            }
                        }
                    }
                ]
            }
        }
    }
    return search_body

@router.post("/video_name", response_model=SearchResponse)
async def search_video_name(request: SearchRequest):
    try:
        query_text = request.value
        top_k = request.top_k
        
        #search_body = make_videoname_search_body(query_text, top_k)
        #raw_results = es_client.search_parsed(search_body)
        raw_results = get_group_frames(query_text)
        results = convert_ImageList(raw_results)

        return SearchResponse(
            success=len(results) > 0,
            query=request.value.strip(),
            results=results,
            total_results=len(results),
            message=f"Found {len(results)} results for query: '{request.value.strip()}'"
        )
    except Exception as e:
        # Log the actual error for debugging
        print(f"Error in search_ocr: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

# HUNG ocr
@router.post("/ocr", response_model=SearchResponse)
async def search_ocr(request: SearchRequest, es_client: OCRClientDeps):
    try:
        print("ES OCR index: ", es_client.index_name)
        
        query_text = request.value
        top_k = request.top_k or 50
    
        ocr_search_body = build_frame_agg_query(query_text, page_size=min(1000, max(10, top_k)))

        raw_results = es_client.search_parsed(ocr_search_body, top_k)
        

        # NEW: Use composite-aggregation frame search
        # print(f"QUERY TEXT = {query_text}")
        # print("BEFORE GETTING RESULTS")
        # raw_results = _ocr_frames_by_agg(
        #     query_text=query_text,
        #     top_k=top_k,
        #     # You can expose these as request options later if you like:
        #     min_terms=None,          # None => require ALL tokens (default in search_frames_ocr)
        #     min_rec_conf=0.0,
        #     min_det_score=0.0,
        #     language=None,
        # )
        # print("AFTER GETTING RESULTS")
        # print(f"RAW RESULTS = {raw_results}")   
        results = convert_ImageList(raw_results)

        return SearchResponse(
            success=len(results) > 0,
            query=request.value.strip(),
            results=results,
            total_results=len(results),
            message=f"Found {len(results)} results for query: '{request.value.strip()}'"
        )
    except Exception as e:
        print(f"Error in search_ocr: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


# @router.post("/asr", response_model=SearchResponse)
# async def search_asr(request: SearchRequest, es_client: ASRClientDeps):
#     try:
#         query_text = request.value.strip()
#         top_k = request.top_k

#         search_body = make_asr_search_body(query_text, top_k)

#         raw_results = es_client.search_parsed(search_body)

#         for res in raw_results:

#             video_name = res['video_name']
#             og_text = res['text']

#             target_keyframe = get_estimate_keyframes(og_text,video_name, query_text)

#             # adjust to keyframe info, keep score,video_name the same
#             res['frame_idx'] = target_keyframe['frame_idx']
#             res['image_path'] = target_keyframe['image_path']
#             res['pts_time'] = target_keyframe['pts_time']

#         results = convert_ImageList(raw_results)

#         return SearchResponse(
#             success=True,
#             query=request.value.strip(),
#             results=results,
#             total_results=len(results),
#             message=f"Found {len(results)} results for query: '{request.value.strip()}'"
#         )
#     except Exception as e:
#         # Log the actual error for debugging
#         print(f"Error in search_asr: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")