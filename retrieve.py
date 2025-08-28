
import faiss
import os
import torch
import clip
from pydantic import BaseModel
import json
import sys
# Define ImageResult locally to avoid import issues
class ImageResult(BaseModel):
    video_name: str
    frame_idx: int
    image_path: str
    score: float

import numpy as np



def process_feat(feat):
    feat = feat.cpu().numpy().astype(np.float32)  # ensure float32 before normalize
    feat = feat.reshape(1, -1)  # reshape to 2D first
    faiss.normalize_L2(feat)  # normalize in-place (requires 2D array)
    return feat

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

faiss_save_dir = os.path.join(script_dir, "FaissIndex")

index_save_path = "faiss_index_transnet.bin"
metadata_save_path = "id_to_name_transnet.json"

DATA_SOURCE = '/REAL_DATA/Data/keyframes-transnet'

print("Num GPUS: ", faiss.get_num_gpus())  # should print >= 1 if GPU is active
device = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE: ", device)

model, preprocess = clip.load("ViT-B/32", device=device)

index_path = os.path.join(faiss_save_dir, index_save_path)
metadata_path = os.path.join(faiss_save_dir, metadata_save_path)
print(f"Loading index from: {index_path}")
print(f"Loading metadata from: {metadata_path}")


 # Load index
index = faiss.read_index(index_path)
print(f"FAISS index loaded from {index_path}")

# Load metadata
with open(metadata_path, 'r') as f:
    metadata = json.load(f)
print(f"Metadata loaded from {metadata_path}")

if index is None or metadata is None:
    print("Error: No FAISS index found. Please run indexing.py first to create the index.")
    exit(1)

print(f"Loaded index with {len(metadata.keys())} images")
print(metadata[str(3232)])

def get_text_embedding(text_query: str):
    """Get CLIP embedding for a text query"""
    with torch.no_grad():
        # Tokenize and encode text
        text_tokens = clip.tokenize([text_query]).to(device)
        text_features = model.encode_text(text_tokens)
        text_embedding = process_feat(text_features)
    return text_embedding

def parse_image_path(image_path: str):
    """Parse image path to extract video name, scene index, and frame index"""
    # Expected format: ExtractedFrames/Video_0001/Video_0001_frame_000123.jpg
    filename = os.path.basename(image_path)
    
    video_file = os.path.splitext(filename)[0]


    # Extract video name from directory
    video_dir = os.path.dirname(image_path)
    video_name = os.path.basename(video_dir)

    frame_idx = video_file.split("_")[-1]
    
    return video_name, frame_idx

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
            frame_path = metadata[str(idx+1)]

            video_name, frame_f = frame_path.split("/")
            frame_idx = int(os.path.splitext(frame_f)[0])

            
            image_path = os.path.join(DATA_SOURCE, frame_path) ####
            
            # Convert inner product to similarity score (0-100%)
            # For normalized vectors, inner product ranges from -1 to 1
            # Convert to 0-100% where 1 = 100% similarity, -1 = 0% similarity
            similarity_score = max(0, min(100, (float(distance) + 1) * 50))
            
            result = ImageResult(
                video_name=video_name,
                frame_idx=frame_idx,
                image_path=image_path,
                score=similarity_score  # Now represents similarity percentage (0-100)
            )
            results.append(result)
    

    return results

def postprocess_output(results: list[ImageResult], max_results: int = 5):
    """Post-process and display search results"""
    print(f"\nTop {min(len(results), max_results)} search results:")
    print("-" * 80)
    
    for i, result in enumerate(results[:max_results]):
        print(f"{i+1}. Video: {result.video_name}")
        print(f"   FrameIndex: {result.frame_idx}")
        print(f"   Path: {result.image_path}")
        print(f"   Similarity: {result.score:.2f}%")
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
