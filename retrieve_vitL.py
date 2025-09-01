# searching keyframes using a text query with CLIP model and FAISS for fast similarity search
import open_clip # open_clip must be imported before torch
import torch

import faiss
import os
import json
from pydantic import BaseModel
import numpy as np
from typing import List, Dict, Any



# Fix OpenMP runtime conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print('Torch version:', torch.__version__)
print('OpenCLIP version:', open_clip.__version__)

CONFIG = {
    'root_path': '/kaggle/input/first-batch-keyframes/keyframes',
    'output_dir': 'extracted_features',
    'model_name': 'ViT-L-14',
    'pretrained': 'openai',
    'batch_size': 32, 
    'num_workers': 4,
    'image_size': 224,
}


def process_feat(feat):
    feat = feat.cpu().numpy().astype(np.float32)  # ensure float32 before normalize
    feat = feat.reshape(1, -1)  # reshape to 2D first
    faiss.normalize_L2(feat)  # normalize in-place (requires 2D array)
    return feat

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
faiss_save_dir = os.path.join(script_dir, "FaissIndex")

index_save_path = "faiss_index_vitL.bin"
metadata_save_path = "id_to_name_vitL.json"

DATA_SOURCE = '/REAL_DATA/keyframes_b1/keyframes'

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print("DEVICE: ", device)


###########################
# Load model và preprocessor
model, _, preprocess = open_clip.create_model_and_transforms(
    CONFIG['model_name'], 
    pretrained=CONFIG['pretrained'],
)

model = model.to(device)
model.eval()

# Test model
print(f"\nModel loaded successfully")
print(f"Model device: {next(model.parameters()).device}")

tokenizer = open_clip.get_tokenizer(CONFIG['model_name'])
################

index_path = os.path.join(faiss_save_dir, index_save_path)
metadata_path = os.path.join(faiss_save_dir, metadata_save_path)
print(f"\nLoading index from: {index_path}")
print(f"Loading metadata from: {metadata_path}")

# Load index
index = faiss.read_index(index_path)
print(f"\nFAISS index loaded from {index_path}")

# Load metadata
with open(metadata_path, 'r') as f:
    metadata = json.load(f)
print(f"Metadata loaded from {metadata_path}")

if index is None or metadata is None:
    print("Error: No FAISS index found. Please run indexing.py first to create the index.")
    exit(1)

print(f"\nLoaded index with {len(metadata.keys())} images\n")


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


def search_query(text_query: str, index, metadata, top_k: int = 10) -> List[Dict[str, Any]]:
    """Search for images similar to a text query. Take a query, a FAISS index, metadata, and the number
    of top results to return.
    """
    # Get text embedding by encoding the text query into CLIP embedding
    text_embedding = get_text_embedding(text_query) # ndarray: (1, embedding_dim) float32
    
    # Search in FAISS index by performing a similarity search
    distances, indices = index.search(text_embedding, top_k)

    results = []
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        if str(idx) in metadata.keys():
            frame_path = metadata[str(idx)]
            
            # check file metadata in FaissIndex folder
            _, video_name, frame_f = frame_path.split("/")

            frame_f = os.path.splitext(frame_f)[0] #####
    
            frame_idx = int(frame_f[1:])
            
            #image_path = os.path.join(DATA_SOURCE, frame_path) ####
            image_path = f"{DATA_SOURCE}/{frame_path}"
            
            # Convert inner product to similarity score (0-100%)
            # For normalized vectors, inner product ranges from -1 to 1
            # Convert to 0-100% where 1 = 100% similarity, -1 = 0% similarity
            #similarity_score = max(0, min(100, (float(distance) + 1) * 50))
            similarity_score = distance

            result = {
                'video_name': video_name,
                'frame_idx': frame_idx,
                'image_path': image_path,
                'score': similarity_score
            }
            results.append(result)
    
    return results

def postprocess_output(results: List[Dict[str, Any]], max_results: int = 5):
    """Post-process and display search results"""
    print(f"\nTop {min(len(results), max_results)} search results:")
    print("-" * 80)
    
    for i, result in enumerate(results[:max_results]):
        print(f"{i+1}. Video: {result['video_name']}")
        print(f"   FrameIndex: {result['frame_idx']}")
        print(f"   Path: {result['image_path']}")
        print(f"   Similarity: {result['score']:.2f}%")
        print()
    
    return results[:max_results]

def interactive_search():
    """Interactive search mode"""
    print("\n=== Interactive Image Search ===")
    print("Enter text queries to search for similar images. Type 'quit' to exit.")
    
    while True:
        query = input("\nEnter your search query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            print("Please enter a valid query.")
            continue
        
        print(f"Searching for: '{query}'...")
        
        try:
            # Search for similar images
            results = search_query(query, index, metadata, top_k=10)
            
            if results:
                # Display results
                postprocess_output(results, max_results=5)
            else:
                print("No results found.")
            
        except Exception as e:
            print(f"Error during search: {e}")

def main():
   
    # Start interactive mode
    interactive_search()

if __name__ == "__main__":
    main()
