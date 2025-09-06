from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any

class ImageResult(BaseModel):
    video_name: str
    frame_idx: int
    image_path: str
    pts_time: float
    score: float

class SearchResponse(BaseModel):
    success: bool
    query: Optional[str] = None
    results: List[ImageResult]
    total_results: int
    message: Optional[str] = None
