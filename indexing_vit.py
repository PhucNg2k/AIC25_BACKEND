import numpy as np
import faiss
import os
import json
import sys
from config import CLIP_EMBED_DIM_L, CLIP_EMBED_DIM_H
from pathlib import Path

# --- Add REAL_DATA to sys.path (if you need to import from it) ---
REAL_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "REAL_DATA"))
if REAL_DATA_DIR not in sys.path:
    sys.path.append(REAL_DATA_DIR)

id_to_name = {}
faiss_save_dir = "FaissIndex"

index_save_path = "faiss_index_vitL.bin"
metadata_save_path = "id_to_name_vitL.json"

os.makedirs(faiss_save_dir, exist_ok=True)

def process_feat(feat):
    feat = feat.astype(np.float32)
    feat = feat.reshape(1, -1)
    faiss.normalize_L2(feat)
    return feat

idx_num = 0
index = faiss.IndexFlatIP(CLIP_EMBED_DIM_H)

# --- Updated folder structure ---
batches_H = ["vitH-batch1", "vitH-batch2"]
batches_L = ["vitL-batch1", "vitL-batch2"]
parts   = ["part1", "part2", "part3", "part4", "part5", "part6", "part7"]

for batch in batches_L:
    for part in parts:
        BASE_PATH = os.path.join(REAL_DATA_DIR, batch, part, "extracted_features")
        if Path(BASE_PATH).exists() == False: continue
        ALL_FEAT_FILE = os.path.join(BASE_PATH, "all_features.npy")
        MAPPING_FILE = os.path.join(BASE_PATH, "all_keyframes_mapping.json")

        print(f"Loading: {ALL_FEAT_FILE}")
        feat_data = np.load(ALL_FEAT_FILE)

        with open(MAPPING_FILE, "r") as f:
            keyframe_mapping = json.load(f)

        print(f"Indexing {len(feat_data)} embeddings from {batch}/{part}")
        for idx, embedding in enumerate(feat_data):
            index.add(process_feat(embedding))

            frame_data = keyframe_mapping[idx]
            frame_path = frame_data["image_path"]

            # Keep last 3 levels of the path
            image_path = "/".join(frame_path.split("/")[-3:])
            id_to_name[idx_num] = image_path
            idx_num += 1

def save_index_and_metadata(index, metadata):
    faiss.write_index(index, os.path.join(faiss_save_dir, index_save_path))
    print(f"FAISS index saved to {os.path.join(faiss_save_dir, index_save_path)}")

    with open(os.path.join(faiss_save_dir, metadata_save_path), "w") as f:
        json.dump(metadata, f)
    print(f"Metadata saved to {os.path.join(faiss_save_dir, metadata_save_path)}")

save_index_and_metadata(index, id_to_name)
