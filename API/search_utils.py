import httpx
from typing import Any, Dict, List, Optional
from od_sqlite import ODSearcher
import os, sys
from pathlib import Path

# --- Localized OD search ---
# OD_DB_ROOT = os.getenv('OD_DB_ROOT', './OD')   # root with *.sqlite shards
OD_GRID_ROWS = int(os.getenv('OD_GRID_ROWS', '8'))
OD_GRID_COLS = int(os.getenv('OD_GRID_COLS', '8'))

BASE = Path(__file__).resolve().parent.parent.parent   # this is VBS_system

OD_DB_ROOT = BASE / "REAL_DATA" / "OD"
sys.path.append(BASE)

# OD + localized search
_searcher = None
def _get_searcher() -> ODSearcher:
    global _searcher
    if _searcher is None:
        print(f"BASE = {BASE}")
        print(f"OD_DB_ROOT = {(OD_DB_ROOT)}", Path(OD_DB_ROOT).exists())
        _searcher = ODSearcher(OD_DB_ROOT, grid_rows=OD_GRID_ROWS, grid_cols=OD_GRID_COLS)
    return _searcher

async def call_od_search(obj_mask: Dict[str, Dict], top_k: int):
    searcher = _get_searcher()
    print(f"OBJ MASK = {obj_mask}")
    results = searcher.search(obj_mask or {})
    print(results)
    results.sort(key=lambda r: r.get('score', 0.0), reverse=True)
    return results[: int(top_k) if top_k else 100]

# helper function
async def post_json(url: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        if response.status_code != 200:
            return None
        return response.json()


async def call_text_search(value: str, top_k: int) -> List[dict]:
    payload = {"value": value, "top_k": top_k}
    data = await post_json("http://localhost:8000/search/text", payload)
    return data.get("results", []) if data else []


async def call_ocr_search(value: str, top_k: int) -> List[dict]:
    payload = {"value": value, "top_k": top_k}
    data = await post_json("http://localhost:8000/es-search/ocr", payload)
    return data.get("results", []) if data else []


async def call_asr_search(value: str, top_k: int) -> List[dict]:
    # Placeholder mapping: reuse video_name search if ASR endpoint not available
    payload = {"value": value, "top_k": top_k}
    data = await post_json("http://localhost:8000/es-search/asr", payload)
    return data.get("results", []) if data else []


async def call_image_search(image_file: Any, top_k: int) -> List[dict]:
    async with httpx.AsyncClient() as client:
        files = {
            "image_file": (
                getattr(image_file, "filename", None) or "upload",
                getattr(image_file, "file", image_file),
                getattr(image_file, "content_type", "application/octet-stream"),
            )
        }
        data = {"top_k": str(top_k)}
        response = await client.post("http://localhost:8000/search/image", files=files, data=data)
        if response.status_code != 200:
            return []
        return response.json().get("results", [])