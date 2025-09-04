from pydantic import BaseModel
from typing import Dict, List, Optional


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class ClassMask(BaseModel):
    n_count: int
    bbox: List[BBox]

class ModalityPayload(BaseModel):
    value: str
    obj_mask: Optional[Dict[str, ClassMask]] = None


class StageModalities(BaseModel):
    text: Optional[ModalityPayload] = None
    ocr: Optional[ModalityPayload] = None
    asr: Optional[ModalityPayload] = None
    img: Optional[ModalityPayload] = None
    localized: Optional[ModalityPayload] = None


class SearchEntryRequest(BaseModel):
    stage_list: Dict[str, StageModalities]
    top_k: int = 100


