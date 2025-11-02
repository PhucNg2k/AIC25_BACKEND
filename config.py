import os

# Resolve workspace root (repo root) and place QUERY_SOURCE under it
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(BACKEND_DIR)

DATA_SOURCE = '/REAL_DATA/keyframes_beit3/keyframes'

QUERY_SOURCE = os.path.join(WORKSPACE_ROOT, 'query-p2-groupA')

ASR_EMBED_MODEL = "intfloat/multilingual-e5-base"

FAISS_SAVE_DIR  = "FaissIndex" 
INDEX_SAVE_PATH = "faiss_index_vitH.bin"
METADATA_AVE_PATH = "id_to_name_vitH.json"

CLIP_EMBED_DIM = 1024

LLM_MODEL = "gemini-2.5-flash"
OCR_INDEX_NAME='ocr_index_v2'
ASR_INDEX_NAME = 'asr_index_chunked'

