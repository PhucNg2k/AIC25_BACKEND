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
from group_utils import get_group_frames
from load_embed_model import get_asr_embedding
from asr_tools import get_estimate_keyframes

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


def make_asr_search_body_2(query_text, top_k=50, fuzziness="AUTO"):
    search_body = {
        "size": top_k,
        "_source": ["video_name", "text"],
        "query": {
            "match": {
                "text": {
                    "query": query_text,
                    "fuzziness": fuzziness
                }
            }
        }
    }
    return search_body

def make_asr_search_body_3(query_text, top_k=50, fuzziness="AUTO"):
    query_vector = get_asr_embedding(query_text)
    num_candidates = max(top_k * 4, 200)

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
                                "boost": 2.0
                            }
                        }
                    },
                    {"match_all": {}}
                ]
            }
        },
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": num_candidates
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


@router.post("/asr", response_model=SearchResponse)
async def search_asr(request: SearchRequest, es_client: ASRClientDeps):
    try:
        query_text = request.value.strip()
        top_k = request.top_k
        
        search_body = make_asr_search_body_2(query_text, top_k)

        raw_results = es_client.search_parsed(search_body)
        
        for res in raw_results:
            
            video_name = res['video_name']
            og_text = res['text']
            
            target_keyframe = get_estimate_keyframes(og_text,video_name, query_text)
            
            # adjust to keyframe info, keep score,video_name the same
            res['frame_idx'] = target_keyframe['frame_idx']
            res['image_path'] = target_keyframe['image_path']
            res['pts_time'] = target_keyframe['pts_time']
            
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
        print(f"Error in search_asr: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")