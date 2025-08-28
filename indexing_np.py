import numpy as np
import faiss
import os
import json

id_to_name = {} 

faiss_save_dir  = "FaissIndex" 
index_save_path = "faiss_index_transnet.bin"
metadata_save_path = "id_to_name_transnet.json"

# Create faiss save directory if it doesn't exist
os.makedirs(faiss_save_dir, exist_ok=True)

DATA_FOLDER = "../REAL_DATA/Data"
target_folder = "clip-features-32-transnet"
target_path = os.path.join(DATA_FOLDER, target_folder)


def process_feat(feat):
    feat = feat.astype(np.float32)  # ensure float32 before normalize
    feat = feat.reshape(1, -1)  # reshape to 2D first
    faiss.normalize_L2(feat)  # normalize in-place (requires 2D array)
    return feat

idx_num = 0

index = faiss.IndexFlatIP(512)  # Inner Product for cosine similarity


for feat_file in sorted(os.listdir(target_path)):
    video_name = os.path.splitext(feat_file)[0]

    clip_feats = np.load(os.path.join(target_path, feat_file))
    
    for frame_idx, embedding in enumerate(clip_feats): # (num_frames, 512)
        # check if frame path exist
        #target_frame = 1 if frame_idx==0 else frame_idx
        target_frame = frame_idx + 1

        index.add(process_feat(embedding))  # Process and add embedding to FAISS
        id_to_name[idx_num] = f"{video_name}/{target_frame:03}.jpg" # L21_V001/001.jpg
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