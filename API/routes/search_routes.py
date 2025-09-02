from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional, Annotated
import sys
import os
from PIL import Image
import io

# Parent directory to import retrieve module, make script-friendly
API_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(API_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from retrieve_vitL import clip_faiss_search
from models import *
from dependecies import search_resource_Deps
from utils import convert_ImageList


router = APIRouter(prefix="/search", tags=["search"])


@router.post("/text", response_model=SearchResponse)
async def text_search(request: SearchRequest, search_provider: search_resource_Deps):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty")
    
    index = search_provider["index"]
    metadata = search_provider["metadata"]

    try:
        en_query = request.query.strip().lower()
        raw_results = clip_faiss_search(en_query, index, metadata, top_k=request.top_k)
        
        results = convert_ImageList(raw_results)

        return SearchResponse(
            success=True,
            query=request.query.strip(),
            results=results,
            total_results=len(results),
            message=f"Found {len(results)} results for query: '{request.query.strip()}'"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error during search: {str(e)}")


@router.post("/image", response_model=SearchResponse)
async def process_image(
    image_file: Annotated[UploadFile, File(...)],
    top_k: Annotated[int, Form()] = 30,
    search_provider: search_resource_Deps = None,
):
    file_name = image_file.filename
    content_type = image_file.content_type

    index = search_provider["index"]
    metadata = search_provider["metadata"]
    
    # Check if it's a valid image type
    if not content_type or not content_type.lower().startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_content = await image_file.read()
        image_PIL = Image.open(io.BytesIO(image_content))
        
        # Convert palette images with transparency to RGBA to avoid warnings
        if image_PIL.mode in ('P', 'PA') and 'transparency' in image_PIL.info:
            image_PIL = image_PIL.convert('RGBA')
        elif image_PIL.mode == 'P':
            image_PIL = image_PIL.convert('RGB')
        elif image_PIL.mode in ('RGBA', 'LA') or 'transparency' in image_PIL.info:
            image_PIL = image_PIL.convert('RGBA')
        elif image_PIL.mode != 'RGB':
            image_PIL = image_PIL.convert('RGB')
        
        raw_results = clip_faiss_search(image_PIL, index, metadata, top_k=top_k)
        
        # Convert raw results to ImageResult instances
        results = convert_ImageList(raw_results)

        return SearchResponse(
            success=True,
            query=file_name, 
            results=results,
            total_results=len(results),
            message=f"Found {len(results)} results for image '{file_name}'"
        )

        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
