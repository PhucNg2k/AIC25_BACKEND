from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
from fastapi import UploadFile



class SearchRequest(BaseModel):
    value: str
    top_k: int



class SearchStage(BaseModel):
    text: Optional[str] = None
    img: Optional[UploadFile] = None
    ocr: Optional[str] = None
    localized: Optional[str] = None

####
class SearchRequestEntry(BaseModel):
    stage_list: Dict[str, SearchStage]
    top_k: int = 100





