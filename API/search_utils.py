import httpx
from typing import Any, Dict, List, Optional


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
    data = await post_json("http://localhost:8000/es-search/video_name", payload)
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