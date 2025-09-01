import os
import sys

# Resolve paths relative to the API directory
API_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(os.path.abspath(API_DIR))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from fastapi import APIRouter, HTTPException

import csv
import json
from csv_models import *
import shutil

from utils import parse_frame_file



SUBMIT_FOLER = "Results"

CSV_DIR = os.path.join(ROOT_DIR, SUBMIT_FOLER, "submission")
CSV_MAPPING = os.path.join(ROOT_DIR, SUBMIT_FOLER, "csv_mapping.json")

if os.path.exists(CSV_DIR):
    shutil.rmtree(CSV_DIR)

os.makedirs(CSV_DIR, exist_ok=True)
with open(CSV_MAPPING, 'w') as f:
    json.dump({}, f, indent=4)

router = APIRouter(prefix="/submitCSV", tags=["submitCSV"])

def check_query_id_exists(query_id_key: str) -> bool:
    if os.path.exists(CSV_MAPPING):
        try:
            with open(CSV_MAPPING, 'r', encoding='utf-8') as f:
                existing_mapping = json.load(f)
                return query_id_key in existing_mapping
        except (json.JSONDecodeError, FileNotFoundError):
            return False
    return False


def add_new_csv_mapping(new_mapping: dict):
    existing_mapping = {}
    if os.path.exists(CSV_MAPPING):
        try:
            with open(CSV_MAPPING, 'r', encoding='utf-8') as f:
                existing_mapping = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_mapping = {}
    existing_mapping.update(new_mapping)
    with open(CSV_MAPPING, 'w', encoding='utf-8') as f:
        json.dump(existing_mapping, f, indent=2, ensure_ascii=False)


def _sanitize_filename(name: str) -> str:
    # remove extension if provided
    base = name.strip()
    if base.lower().endswith('.csv'):
        base = base[:-4]
    # replace spaces with dashes
    base = base.replace(' ', '-')
    # allow alnum, dash, underscore, dot
    safe = ''.join(ch for ch in base if ch.isalnum() or ch in ['-', '_', '.'])
    return safe or 'submission'


@router.post("/kis", response_model=MakeCSV_Response)
async def make_csv(request: KIS_rq_CSV):
    try:
        query_id = request.query_id.strip()
        query_str = request.query_str
        selected_frames = request.selected_frames

        id_key = query_id
        csv_filename = f"{_sanitize_filename(query_id)}.csv"
        csv_filepath = os.path.join(CSV_DIR, csv_filename)

        if check_query_id_exists(id_key):
            raise HTTPException(
                status_code=409,
                detail=f"CSV for query_id {query_id} already exists. Cannot overwrite existing submissions."
            )

        if os.path.exists(csv_filepath):
            raise HTTPException(
                status_code=409,
                detail=f"CSV file {csv_filename} already exists. Cannot overwrite existing submissions."
            )

        csv_data = []
        for frameData in selected_frames:
            video_name, frame_index = parse_frame_file(frameData)
            frame_index = int(frame_index)
            csv_data.append([video_name, frame_index])

        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
            writer.writerows(csv_data)

        mapping_update = {id_key: query_str}
        add_new_csv_mapping(mapping_update)

        return MakeCSV_Response(
            success=True,
            message=f"CSV file created successfully: {csv_filename}",
            query_id=query_id,
            csv_file=csv_filename,
            total_frames=len(selected_frames)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating CSV: {str(e)}")


@router.post("/qa", response_model=MakeCSV_Response)
async def make_qa_csv(request: QA_CSV_Request):
    try:
        query_id = request.query_id.strip()
        query_str = request.query_str
        qa_data = request.qa_data or []

        if len(qa_data) == 0:
            raise HTTPException(status_code=400, detail="qa_data cannot be empty")

        id_key = query_id
        csv_filename = f"{_sanitize_filename(query_id)}.csv"
        csv_filepath = os.path.join(CSV_DIR, csv_filename)

        if check_query_id_exists(id_key):
            raise HTTPException(
                status_code=409,
                detail=f"CSV for query_id {query_id} already exists. Cannot overwrite existing submissions."
            )

        if os.path.exists(csv_filepath):
            raise HTTPException(
                status_code=409,
                detail=f"CSV file {csv_filename} already exists. Cannot overwrite existing submissions."
            )

        total_rows = 0
        # Manually compose to ensure only the answer is quoted and always wrapped in quotes
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            for item in qa_data:
                video_name = str(item.video_name).strip()
                frame_index = int(item.frame_idx)
                raw_answer = "" if item.answer is None else str(item.answer)
                answer = raw_answer.strip()
                # If already wrapped in one pair of quotes, remove them first
                if len(answer) >= 2 and answer.startswith('"') and answer.endswith('"'):
                    answer = answer[1:-1]
                # Escape internal quotes by doubling
                answer_escaped = answer.replace('"', '""')
                # Always wrap the answer in quotes
                line = f"{video_name},{frame_index},\"{answer_escaped}\"\n"
                csvfile.write(line)
                total_rows += 1

        mapping_update = {id_key: query_str}
        add_new_csv_mapping(mapping_update)

        return MakeCSV_Response(
            success=True,
            message=f"CSV file created successfully: {csv_filename}",
            query_id=query_id,
            csv_file=csv_filename,
            total_frames=total_rows
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating QA CSV: {str(e)}")


@router.post("/trake", response_model=MakeCSV_Response)
async def make_trake_csv(request: TRAKE_CSV_Request):
    try:
        query_id = request.query_id.strip()
        query_str = request.query_str
        trake_data = request.trake_data or []

        if len(trake_data) == 0:
            raise HTTPException(status_code=400, detail="trake_data cannot be empty")

        id_key = query_id
        csv_filename = f"{_sanitize_filename(query_id)}.csv"
        csv_filepath = os.path.join(CSV_DIR, csv_filename)

        if check_query_id_exists(id_key):
            raise HTTPException(
                status_code=409,
                detail=f"CSV for query_id {query_id} already exists. Cannot overwrite existing submissions."
            )

        if os.path.exists(csv_filepath):
            raise HTTPException(
                status_code=409,
                detail=f"CSV file {csv_filename} already exists. Cannot overwrite existing submissions."
            )

        total_rows = 0
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
            for entry in trake_data:
                row = [str(entry.video_name)] + [int(fid) for fid in entry.frames]
                writer.writerow(row)
                total_rows += 1

        mapping_update = {id_key: query_str}
        add_new_csv_mapping(mapping_update)

        return MakeCSV_Response(
            success=True,
            message=f"CSV file created successfully: {csv_filename}",
            query_id=query_id,
            csv_file=csv_filename,
            total_frames=total_rows
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating TRAKE CSV: {str(e)}")


