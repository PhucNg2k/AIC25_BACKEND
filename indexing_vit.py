import numpy as np
import faiss
import os
import json
from config import CLIP_EMBED_DIM



id_to_name = {} 
faiss_save_dir  = "FaissIndex" 

index_save_path = "faiss_index_vitL.bin"
metadata_save_path = "id_to_name_vitL.json"

# Create faiss save directory if it doesn't exist
os.makedirs(faiss_save_dir, exist_ok=True)

def process_feat(feat):
    feat = feat.astype(np.float32)  # ensure float32 before normalize
    feat = feat.reshape(1, -1)  # reshape to 2D first
    faiss.normalize_L2(feat)  # normalize in-place (requires 2D array)
    return feat

idx_num = 0
index = faiss.IndexFlatIP(CLIP_EMBED_DIM)  # Inner Product for cosine similarity

taget_folder = ['vit-batch1', 'vit-batch2']

for folder in taget_folder:
    BASE_PATH = f"../REAL_DATA/{folder}/extracted_features"

    ALL_FEAT_FILE = os.path.join(BASE_PATH, "all_features.npy" )

    feat_data = np.load(ALL_FEAT_FILE)

    with open(os.path.join(BASE_PATH, "all_keyframes_mapping.json"), 'r') as f:
        keyframe_mapping = json.load(f) 

    print(folder)
    
    for idx, embedding in enumerate(feat_data):
        index.add(process_feat(embedding))
        
        frame_data = keyframe_mapping[idx]
        frame_path = frame_data['image_path']

        image_path = frame_path.split("/")[-3:]
        image_path = "/".join(image_path)

        id_to_name[idx_num] = image_path
        idx_num+=1


def save_index_and_metadata(index, metadata):
    """Save FAISS index and image metadata"""
    # Save FAISS index
    faiss.write_index(index, os.path.join(faiss_save_dir,index_save_path))
    print(f"FAISS index saved to {os.path.join(faiss_save_dir,index_save_path)}")
    
    with open(os.path.join(faiss_save_dir, metadata_save_path), 'w') as f:
        json.dump(metadata, f)
    print(f"Metadata saved to {os.path.join(faiss_save_dir, metadata_save_path)}")


save_index_and_metadata(index, id_to_name)