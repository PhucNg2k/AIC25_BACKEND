from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import sys
import json
from typing import List, Dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from API.ElasticSearch.ESclient import ASRClient


ASR_DATA = "../../REAL_DATA/asr_chunked"


def load_data(asr_dir: str):
    features = np.load(os.path.join(asr_dir, 'asr_full_embedding.npy'))
    with open(os.path.join(asr_dir, 'asr_full_mapping.json'), 'r') as f:
        mapping = json.load(f)
    return features, mapping


def build_documents(embeddings: np.ndarray, mapping: Dict[str, Dict]) -> List[Dict]:
    documents: List[Dict] = []
    for idx, embedding in enumerate(embeddings):
        value = mapping.get(str(idx))
        if value is None:
            continue
        video_name = value.get('video_name')
        text_content = value.get('text')
        if video_name is None:
            continue
            
        # ensure JSON serializable list
        if isinstance(embedding, np.ndarray):
            emb_list = embedding.tolist()
        elif isinstance(embedding, list):
            emb_list = embedding
        else:
            emb_list = embedding.tolist()
        
        documents.append({
            'video_name': str(video_name),
            'text': str(text_content) if text_content is not None else "",
            'embedding': emb_list,
        })
    return documents


def main():
    load_dotenv()
    es_url = os.getenv("ES_LOCAL_URL")
    es_api_key = os.getenv("ES_LOCAL_API_KEY")
    index_name = os.getenv("ASR_INDEX_NAME", "asr_index_chunked")

    embeddings, mapping = load_data(ASR_DATA)
    embedding_dims = int(embeddings.shape[1]) if len(embeddings.shape) == 2 else 768

    client = ASRClient(
        hosts=[es_url],
        api_key=es_api_key,
        index_name=index_name,
        embedding_dims=embedding_dims,
    )

    client.create_index(force=True)

    docs = build_documents(embeddings, mapping)
    # index in batches using client's bulk_index
    client.bulk_index(docs, batch_size=1000)


if __name__ == "__main__":
    main()