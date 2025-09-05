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
    'root_path': '/kaggle/input/second-batch-keyframes/Keyframes_Batch_2',
    'output_dir': 'extracted_features',
    'model_name': 'ViT-H-14',
    'pretrained': 'laion2b_s32b_b79k',
    'batch_size': 32, 
    'num_workers': 4,
    'image_size': 224,
}


def load_index():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    faiss_save_dir = os.path.join(script_dir, "FaissIndex")

    index_save_path = "faiss_index_vitL.bin"
    metadata_save_path = "id_to_name_vitL.json"

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

    return model, tokenizer, preprocess, index, metadata, device

model, tokenizer, preprocess, index, metadata, device = load_index()

