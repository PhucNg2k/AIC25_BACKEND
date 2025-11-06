import os

# Resolve workspace root (repo root) and place QUERY_SOURCE under it
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.dirname(BACKEND_DIR)

DATA_SOURCE = '/REAL_DATA/Data/keyframes_beit3'

QUERY_SOURCE = os.path.join(WORKSPACE_ROOT, 'query-p3-groupA')
ASR_EMBED_MODEL = "intfloat/multilingual-e5-base"

FAISS_SAVE_DIR  = "FaissIndex" 
INDEX_SAVE_PATH = "faiss_index_vith_1.bin"
METADATA_AVE_PATH = "id_to_name_vith_1.json"


LLM_MODEL = "gemini-2.5-flash"
OCR_INDEX_NAME='ocr_index_v2'
ASR_INDEX_NAME = 'asr_index_chunked'
CLIP_EMBED_DIM_L = 768
CLIP_EMBED_DIM_H = 1024
