import open_clip
import torch
import os
import json
import faiss
from pydantic import BaseModel
import numpy as np

# Fix OpenMP runtime conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class ImageResult(BaseModel):
    video_name: str
    frame_idx: int
    image_path: str
    score: float
    
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

faiss_save_dir = os.path.join(script_dir, "FaissIndex")

index_save_path = "faiss_index_vitL.bin"
metadata_save_path = "id_to_name_vitL.json"

DATA_SOURCE = '/REAL_DATA/keyframes_b1/keyframes'


CONFIG = {
    'root_path': '/kaggle/input/first-batch-keyframes/keyframes',
    'output_dir': 'extracted_features',
    'model_name': 'ViT-L-14',
    'pretrained': 'openai',
    'batch_size': 32, 
    'num_workers': 4,
    'image_size': 224,
}

# Safe GPU detection for MPS compatibility
try:
    if torch.cuda.is_available():
        device = 'cuda'
        print("Num GPUS: ", faiss.get_num_gpus())  # Only check FAISS GPUs for CUDA
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Apple Silicon)")
    else:
        device = 'cpu'
        print("Using CPU")
except Exception as e:
    print(f"GPU detection error: {e}")
    device = 'cpu'
    print("Falling back to CPU")

print("DEVICE: ", device)


print(f"Loading {CONFIG['model_name']} model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    CONFIG['model_name'], 
    pretrained=CONFIG['pretrained'],
)


# Move model to device with error handling
try:
    model = model.to(device)
    model.eval()
    print(f"Model moved to {device}")
except Exception as device_error:
    print(f"Error moving model to {device}: {device_error}")
    print("Falling back to CPU")
    device = 'cpu'
    model = model.to(device)
    model.eval()

# Verify model is on correct device
print(f"Model loaded successfully")
print(f"Model device: {next(model.parameters()).device}")


tokenizer = open_clip.get_tokenizer(CONFIG['model_name'])
print("Tokenizer initiaized")


index_path = os.path.join(faiss_save_dir, index_save_path)
metadata_path = os.path.join(faiss_save_dir, metadata_save_path)
print(f"Loading index from: {index_path}")
print(f"Loading metadata from: {metadata_path}")



index = faiss.read_index(index_path)
print(f"FAISS index loaded from {index_path}")

with open(metadata_path, 'r') as f:
        metadata = json.load(f)
print(f"Metadata loaded from {metadata_path}")


def process_feat(feat):
    feat = feat.cpu().numpy().astype(np.float32)  # ensure float32 before normalize
    feat = feat.reshape(1, -1)  # reshape to 2D first
    faiss.normalize_L2(feat)  # normalize in-place (requires 2D array)
    return feat

def get_text_embedding(text_query: str):
    """Get CLIP embedding for a text query"""
    with torch.no_grad():
        # Tokenize and move to device
        text_tokens = tokenizer([text_query], context_length=model.context_length).to(device)
        # Encode
        text_features = model.encode_text(text_tokens)
        # Normalize and return as numpy for FAISS
        text_embedding = process_feat(text_features)
        
    return text_embedding

def search_query(text_query: str, index, metadata, top_k: int = 10) -> list[ImageResult]:
    """Search for images similar to a text query"""
    # Get text embedding
    text_embedding = get_text_embedding(text_query)
    
    # Search in FAISS index
    distances, indices = index.search(text_embedding, top_k)

    # Convert results to ImageResult objects

    results = []
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        if str(idx) in metadata.keys():
            frame_path = metadata[str(idx)]
            
            _, video_name, frame_f = frame_path.split("/")

            frame_f = os.path.splitext(frame_f)[0] #####
    
            frame_idx = int(frame_f[1:])
            
            image_path = os.path.join(DATA_SOURCE, frame_path) ####
            
            #similarity_score = max(0, min(100, (float(distance) + 1) * 50))
            similarity_score = distance

            result = ImageResult(
                video_name=video_name,
                frame_idx=frame_idx,
                image_path=image_path,
                score=similarity_score  # Now represents similarity percentage (0-100)
            )
            results.append(result)
    

    return results

query='fire'
results = search_query(query, index, metadata, top_k=10)
print(results)