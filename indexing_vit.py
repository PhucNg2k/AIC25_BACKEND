import numpy as np
import faiss
import os
import json
from config import CLIP_EMBED_DIM



id_to_name = {} 
faiss_save_dir  = "FaissIndex" 

index_save_path = "faiss_index_vitH.bin"
metadata_save_path = "id_to_name_vitH.json"

# Create faiss save directory if it doesn't exist
os.makedirs(faiss_save_dir, exist_ok=True)

def process_feat(feat):
    feat = feat.astype(np.float32)  # ensure float32 before normalize
    feat = feat.reshape(1, -1)  # reshape to 2D first
    faiss.normalize_L2(feat)  # normalize in-place (requires 2D array)
    return feat

idx_num = 0
index = faiss.IndexFlatIP(CLIP_EMBED_DIM)  # Inner Product for cosine similarity

target_folder = ['vitH-batch1', 'vitH-batch2']

for folder in target_folder:
    BASE_PATH = f"../REAL_DATA/{folder}/extracted_features"

    ALL_FEAT_FILE = os.path.join(BASE_PATH, "all_features.npy" )

    feat_data = np.load(ALL_FEAT_FILE)
    # print(type(feat_data)) # <class 'numpy.ndarray'>
    # print(feat_data.shape) # (223024, 1024) batch1
    # print(feat_data[0])
    
    with open(os.path.join(BASE_PATH, "all_keyframes_mapping.json"), 'r') as f:
        # List[Dict] 
        keyframe_mapping = json.load(f) 
    # print(type(keyframe_mapping)) # <class 'list'>
    # print(len(keyframe_mapping)) # 223024
    # print(type(keyframe_mapping[0])) # <class 'dict'>
    # print(keyframe_mapping[0].keys()) # dict_keys(['video_id', 'frame_id', 'frame_number', 'image_path'])
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
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to {os.path.join(faiss_save_dir, metadata_save_path)}")


save_index_and_metadata(index, id_to_name)
