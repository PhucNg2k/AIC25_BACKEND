from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
from fastapi import UploadFile

####
class SearchRequestEntry(BaseModel):
    text: Optional[str] = None
    img: Optional[UploadFile] = None
    ocr: Optional[str] = None
    localized: Optional[str] = None
    top_k: int = 100

class RerankRequest(BaseModel):
    chosen_frames: List[str]
    query: Optional[str] = None
    top_k: int = 50

### 

class ImageResult(BaseModel):
    video_name: str
    frame_idx: int
    image_path: str
    pts_time: float
    score: float

####
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


class SearchResponse(BaseModel):
    success: bool
    query: Optional[str] = None
    results: List[ImageResult]
    total_results: int
    message: Optional[str] = None

