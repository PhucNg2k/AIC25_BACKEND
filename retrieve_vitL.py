# searching keyframes using a text query with CLIP model and FAISS for fast similarity search

import torch
import faiss
import os
import json
import io
from pydantic import BaseModel
from typing import List, Dict, Any, Union

from config import DATA_SOURCE
from model_loading import model, tokenizer, preprocess, index, metadata, device

from API.frame_utils import get_metakey, get_pts_time, get_frame_path


from PIL import Image
import numpy as np

def process_feat(feat):
    feat = feat.cpu().numpy().astype(np.float32)  # ensure float32 before normalize
    feat = feat.reshape(1, -1)  # reshape to 2D first
    faiss.normalize_L2(feat)  # normalize in-place (requires 2D array)
    return feat


def resize_image(image, target_size=(224, 224)):
    """Resize image to target size using PIL"""
    
    # Convert to PIL Image if needed
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise ValueError(f"Unsupported image type: {type(image)}")
        
    resized_image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    return resized_image

def get_image_embedding(image: Image.Image):
    # Ensure PIL Image and apply the same preprocess used during training
    pil_img = resize_image(image)
    input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(input_tensor)
        image_embedding = process_feat(image_features)
        
    return image_embedding    

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


def clip_faiss_search(query: Union[str, Image.Image], index, metadata, top_k: int = 10) -> List[Dict[str, Any]]:
    
    # Get text embedding by encoding the text query into CLIP embedding
    if isinstance(query, str): # -> text query
        embedding = get_text_embedding(query) # ndarray: (1, embedding_dim) float32
    else:
        embedding = get_image_embedding(query) # -> image query

    # Search in FAISS index by performing a similarity search
    """
    distances: NumPy array of shape (nq, k) (nq: number of queries, k: number of nearest neighbors to retrieve (top-k))
    indices: NumPy array of shape (nq, k)
    indices[i, j] is the database ID of the j-th neighbor of query i.
    distances[i, j] is that neighbor’s score under the index’s metric.
    """
    distances, indices = index.search(embedding, top_k)

    results = []
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        if str(idx) in metadata.keys():
            frame_path = metadata[str(idx)] # "Videos_L21_a/L21_V001/f000000.webp"
            
            # check file metadata in FaissIndex folder
            # video_name: L21_V001, frame_f: f000000.webp
            _, video_name, frame_f = frame_path.split("/")

            frame_f = os.path.splitext(frame_f)[0]

            frame_idx = int(frame_f[1:])
            
            image_path = os.path.join(DATA_SOURCE, frame_path) ####
            
            # Convert inner product to similarity score (0-100%)
            # For normalized vectors, inner product ranges from -1 to 1
            # Convert to 0-100% where 1 = 100% similarity, -1 = 0% similarity
            #similarity_score = max(0, min(100, (float(distance) + 1) * 50))

            
            metakey = get_metakey(video_name, frame_idx)
            pts_time = get_pts_time(metakey)
            image_path = get_frame_path(metakey)

            result = {
                'video_name': video_name,
                'frame_idx': frame_idx,
                'image_path': image_path,
                'score': similarity_score,
                'pts_time': pts_time
            }
            results.append(result)
    
    return results




def main():
   def interactive_search():

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
            results = clip_faiss_search(query, index, metadata, top_k=10)
            
            if results:
                # Display results
                postprocess_output(results, max_results=5)
            else:
                print("No results found.")
            
        except Exception as e:
            print(f"Error during search: {e}")

    # Start interactive mode
    interactive_search()

if __name__ == "__main__":
    from model_loading import index, metadata
    main()
