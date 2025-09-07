from pydantic import BaseModel, field_validator
from typing import Dict, List, Optional
import json


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class ClassMask(BaseModel):
    count_condition: str
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
    weight_dict: Dict[str, float]


class SearchEntryRequest(BaseModel):
    stage_list: Dict[str, StageModalities]
    top_k: int = 100

    @field_validator('stage_list', mode='before')
    @classmethod
    def parse_stage_list_from_string(cls, v):
        # Accept already-parsed dict or JSON string from multipart/form-data
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                pass
        return v


