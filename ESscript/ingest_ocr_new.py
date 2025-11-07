# AIC25_BACKEND/ESscript/ingest_ocr_new.py
from dotenv import load_dotenv
import os
import sys
import json
import gzip
import hashlib
import re

from typing import Iterator, Dict, Any, List, Tuple
from elasticsearch.exceptions import RequestError

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

load_dotenv()
es_url = os.getenv("ES_LOCAL_URL")
es_api_key = os.getenv("ES_LOCAL_API_KEY")

# Tune these without editing code
INDEX_NAME   = os.getenv("OCR_INDEX_NAME", "ocr_index_v2")
print(f"INDEX_NAME = {INDEX_NAME}")
DATA_ROOT = os.getenv(
    "OCR_DATA_ROOT",
    os.path.abspath(os.path.join(ROOT_DIR,  "..", "REAL_DATA"))
)
BATCH_DIRS   = os.getenv("OCR_BATCH_DIRS", "ocr_batch1,ocr_batch2").split(",")
BULK_SIZE    = int(os.getenv("OCR_BULK_SIZE", "5000"))

from API.ElasticSearch.ESclient import OCRClient  # uses your existing module layout

# ---------- helpers ----------
def is_jsonl(path: str) -> bool:
    return path.endswith(".jsonl") or path.endswith(".jsonl.gz")

def open_text(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "r", encoding="utf-8")

def safe_int_from_frame_id(frame_id: Any) -> int:
    """
    Extract numeric part from frame ids like 'f000108' -> 108.
    Returns -1 when not found (we keep it as a doc field, parse_hits can ignore).
    """
    try:
        if frame_id is None:
            return -1
        s = str(frame_id)
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else -1
    except Exception:
        return -1

def doc_id(batch: str, rel_path: str, line_no: int, rec: Dict[str, Any]) -> str:
    """
    Stable _id so re-running the ingester won't duplicate docs.
    Hash includes file path + line no + key OCR fields.
    """
    h = hashlib.sha1() # create a new SHA1 hash object (40 character hex string)
    payload = "|".join([
        batch,
        rel_path.replace(os.sep, "/"),
        str(line_no),
        str(rec.get("video_id", "")),
        str(rec.get("frame_id", "")),
        str(rec.get("text", "")),
        str(rec.get("bbox", "")),
    ])
    h.update(payload.encode("utf-8", errors="ignore"))
    return h.hexdigest()

def walk_jsonl_files(root: str, batch_dirs: List[str]) -> Iterator[Tuple[str, str]]:
    """
    Yields (batch_name, file_path)
    """
    for batch in batch_dirs:
        # print(batch)
        base = os.path.join(root, batch.strip())
        if not os.path.exists(base):
            print(f"⚠️ Skipping missing batch: {base}")
            continue
        for dirpath, dirname, filenames in os.walk(base):
            # print(f"dirpath = {dirpath}")
            # print()
            # print(f"dirname = {dirname}")
            # print()
            # print(f"filenames = {filenames}")
            for fn in filenames:
                if is_jsonl(fn):
                    yield batch.strip(), os.path.join(dirpath, fn)
                    # print("IM HERE")
            

def path_meta(batch: str, full_path: str) -> Dict[str, str]:
    """
    Derive inter_folder/leaf_folder/file_name from the path under DATA_ROOT/batch.
    """
    rel_from_root = os.path.relpath(full_path, os.path.join(DATA_ROOT, batch))
    parts = rel_from_root.split(os.sep)
    file_name = parts[-1] # f000000.webp
    leaf_folder = parts[-2] if len(parts) >= 2 else "" # L21_V001
    inter_folder = parts[-3] if len(parts) >= 3 else "" # Videos_L21
    return {
        "batch": batch, # ocr_batch1
        "inter_folder": inter_folder, # Videos_L21
        "leaf_folder": leaf_folder,# L21_V001
        "file_name": file_name, # f000000.webp
        "rel_path": rel_from_root, # Videos_L21/L21_V001/f000000.webp
    }

def record_to_doc(pm: Dict[str, str], rec: Dict[str, Any], line_no: int) -> Dict[str, Any]:
    """
    Transform one OCR json line -> ES document that matches OCRClient mapping.
    Args:
        pm: path_meta (Dict)

    """
    vid = rec.get("video_id")
    fid = rec.get("frame_id")
    frame_idx = safe_int_from_frame_id(fid)

    # Keep video_name as alias to video_id for downstream code compatibility
    doc: Dict[str, Any] = {
        "_id":        doc_id(pm["batch"], pm["rel_path"], line_no, rec),
        "batch":      pm["batch"],
        "inter_folder": pm["inter_folder"],
        "leaf_folder":  pm["leaf_folder"],
        "file_name":    pm["file_name"],

        "video_id":   vid,      # L21_V023
        "video_name": vid,      # important for existing parse pipeline
        "frame_id":   fid,
        "frame_idx":  frame_idx,

        "text_raw":   rec.get("text_raw"),
        "text":       rec.get("text"),
        "text_folded": rec.get("text_folded"),

        "language":   rec.get("language"),
        "source":     rec.get("source"),
        "det_score":  rec.get("det_score"),
        "rec_conf":   rec.get("rec_conf"),
        "bbox":       rec.get("bbox"),
        "poly":       rec.get("poly"),
    }
    return doc

# ---------- main ----------
if __name__ == "__main__":
    ocr_client = OCRClient(
        hosts=[es_url],
        api_key=es_api_key,
        index_name=INDEX_NAME
    )

    # Create index with the richer mapping (safe to force in dev; set FORCE_CREATE=0 to keep)
    force_create = os.getenv("OCR_FORCE_CREATE", "1") == "1"
    ocr_client.create_index(force=force_create)

    buffer: List[Dict[str, Any]] = []
    total_files = 0
    total_docs = 0

    for batch, fpath in walk_jsonl_files(DATA_ROOT + "/OCR_DeepSolo_PARSeq", BATCH_DIRS):
        total_files += 1
        # print(f"fpath= {fpath}")
        
        pm = path_meta(batch, fpath)
        print(f"\n📄 Processing: {pm['rel_path']}")
        # print(pm) 
        
        try:
            with open_text(fpath) as fh:
                for i, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        # skip bad line, but continue
                        continue

                    buffer.append(record_to_doc(pm, rec, i))
                    if len(buffer) >= BULK_SIZE:
                        ocr_client.bulk_index(buffer, batch_size=BULK_SIZE)
                        total_docs += len(buffer)
                        buffer = []
        except Exception as e:
            print(f"❌ Error reading {fpath}: {e}")

        
    # flush remaining
    if buffer:
        ocr_client.bulk_index(buffer, batch_size=BULK_SIZE)
        total_docs += len(buffer)

    print(f"\n✅ Done. Files: {total_files}, Docs indexed: {total_docs}")
