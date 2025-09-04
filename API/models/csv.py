from pydantic import BaseModel
from typing import List

class MakeCSV_Response(BaseModel):
    success: bool
    message: str
    query_id: str
    csv_file: str
    total_frames: int


class KIS_rq_CSV(BaseModel):
    query_id: str
    query_str: str
    selected_frames: List[str]


class QADataItem(BaseModel):
    video_name: str
    frame_idx: int
    answer: str


class QA_CSV_Request(BaseModel):
    query_id: str
    query_str: str
    qa_data: List[QADataItem]


class TrakeVideoEntry(BaseModel):
    video_name: str
    frames: List[int]


class TRAKE_CSV_Request(BaseModel):
    query_id: str
    query_str: str
    trake_data: List[TrakeVideoEntry]
