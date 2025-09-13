from models.request import *
from models.response import *

from fastapi import  HTTPException, APIRouter
import sys
import os
# NEW: bring in the aggregation search + frame utils
from search_frames_ocr import search_frames as agg_search_frames
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

def make_ocr_search_body(query_text, top_k=50, fuzziness="AUTO"):
    """
    Robust OCR search:
      - text.folded with fuzziness (accent-insensitive + tolerant to small OCR errors)
      - text.ngrams fallback for heavier noise
      - text (raw) as a weaker signal
      - confidence-aware ranking via rec_conf / det_score
    """
    # Normalize fuzziness: prefer 1 for small OCR mistakes like extra/missing letter
    fz = 1 if str(fuzziness).upper() == "AUTO" else fuzziness

    search_body = {
        "size": top_k,
        "track_total_hits": True,
        "_source": ["video_folder", "video_name", "frame_id", "text", "rec_conf", "det_score"],
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "should": [
                            # Primary: accent-insensitive, fuzzy lexical match
                            {
                                "match": {
                                    "text.folded": {
                                        "query": query_text,
                                        "fuzziness": fz,
                                        "prefix_length": 0,
                                        "max_expansions": 50,
                                        "boost": 5.0
                                    }
                                }
                            },
                            # Secondary: raw text (in case folded analyzer missed something)
                            {
                                "match": {
                                    "text": {
                                        "query": query_text,
                                        "fuzziness": fz,
                                        "prefix_length": 0,
                                        "max_expansions": 50,
                                        "boost": 2.0
                                    }
                                }
                            },
                            # Safety net: character n-grams for shredded tokens
                            {
                                "match": {
                                    "text.ngrams": {
                                        "query": query_text,
                                        "boost": 1.0
                                    }
                                }
                            }
                        ],
                        "minimum_should_match": 1
                    }
                },
                # Confidence-aware scoring
                "functions": [
                    {
                        "field_value_factor": {
                            "field": "rec_conf",
                            "modifier": "sqrt",
                            "factor": 1.0,
                            "missing": 0.3
                        }
                    },
                    {
                        "field_value_factor": {
                            "field": "det_score",
                            "modifier": "sqrt",
                            "factor": 0.5,
                            "missing": 0.3
                        }
                    }
                ],
                "score_mode": "sum",
                "boost_mode": "sum"
            }
        }
    }
    return search_body

def _ocr_frames_by_agg(
    query_text: str,
    top_k: int = 50,
    min_terms: int = None,
    min_rec_conf: float = 0.0,
    min_det_score: float = 0.0,
    language: str = None,
) -> list[dict]:
    """
    Calls the composite-aggregation search and adapts each frame to the UI shape
    your convert_ImageList() already understands.
    """
    # Keep your index selection consistent with your system env/config
    index_name = os.getenv("OCR_INDEX_NAME", "ocr_index_v2")  # align with your mapping/index
    print(f"INDEX_NAME = {index_name}")
    rows = agg_search_frames(
        query_text,
        index_name=index_name,
        min_terms=min_terms,          # default: require ALL tokens (search_frames_ocr behavior)
        use_folded=True,              # accent-insensitive
        top_n=top_k,
        min_rec_conf=min_rec_conf,
        min_det_score=min_det_score,
        language=language,
    )
    print(f"LEN ROWS = {len(rows)}")
    out: list[dict] = []
    for r in rows:
        video_name = r["video_name"]
        # print(f"video name = {video_name}")
        # search_frames_ocr returns "frame_id" as string; we convert to int idx for your UI
        try:
            frame_idx = int(float(r["frame_id"][1:]))
        except Exception:
            continue
        # print(f"frame_idx = {frame_idx}")
        metakey  = get_metakey(video_name, frame_idx)
        image    = get_frame_path(metakey)
        pts_time = get_pts_time(metakey)

        # Choose a stable score for sorting; sum_rec_conf works well. You could
        # also combine matched_terms and confidences if you want.
        out.append({
            "video_name": video_name,
            "frame_idx": frame_idx,
            "score": r.get("sum_rec_conf", 0.0),
            "image_path": image,
            "pts_time": pts_time,
            # If you want to surface debug info in UI later:
            # "matched_terms": r.get("matched_terms"),
            # "max_rec_conf": r.get("max_rec_conf"),
            # "examples": r.get("examples", []),
        })
        
    print(f"LEN OUT = {len(out)}")
    return out

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

        

# @router.post("/ocr", response_model=SearchResponse)
# async def search_ocr(request: SearchRequest, es_client: OCRClientDeps):
#     try:
#         query_text = request.value
#         top_k = request.top_k
        
#         search_body = make_ocr_search_body(query_text, top_k)

#         raw_results = es_client.search_parsed(search_body)

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
#         print(f"Error in search_ocr: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@router.post("/ocr", response_model=SearchResponse)
async def search_ocr(request: SearchRequest, es_client: OCRClientDeps):
    print("CALL OCR")
    try:
        query_text = request.value
        top_k = request.top_k or 50

        # NEW: Use composite-aggregation frame search
        # print(f"QUERY TEXT = {query_text}")
        # print("BEFORE GETTING RESULTS")
        raw_results = _ocr_frames_by_agg(
            query_text=query_text,
            top_k=top_k,
            # You can expose these as request options later if you like:
            min_terms=None,          # None => require ALL tokens (default in search_frames_ocr)
            min_rec_conf=0.0,
            min_det_score=0.0,
            language=None,
        )
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
        
#         search_body = make_asr_search_body_2(query_text, top_k)

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