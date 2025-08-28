

import faiss
import torch
from PIL import Image
import clip  # OpenAI CLIP or openclip
import numpy as np
import os


class ClipRetrieval:
    def __init__(self, model_name="ViT-B/32", device="cuda"):
        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.device = device

        # FAISS index & metadata
        self.index = None
        self.video_metadata = None
        

    # ---- Embeddings ----
    def compute_text_embedding(self, text_query: str):
        with torch.no_grad():
            text_tokens = clip.tokenize([text_query]).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            return text_features.cpu().numpy()

    def compute_image_embedding(self, image: Image.Image):
        with torch.no_grad():
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(image_input)
            return image_features.cpu().numpy()

    # ---- Shot utils ----
    
    def get_frame_path(self, frame_index):
        pass

    def open_image(self, frame_path):
        if os.path.exists(frame_path):
            return Image.open(frame_path)
            
    def get_shot_boundaries(self, frame_index):
        sec_interval = 1 # +-1 second
        vid_fps = 30
        frames_interval = sec_interval * vid_fps
        return (frame_index-frame_index, frame_index+frames_interval)
    
    def sample_frames_index(self, shot_bound, n_frames=5):
        start_idx, end_idx = shot_bound
        total_frames = end_idx - start_idx + 1

        if total_frames <= 0 or n_frames <= 0:
            return []

        # If only one frame requested, return the middle one
        if n_frames == 1:
            return [start_idx + total_frames // 2]

        # If fewer frames than requested, return all available indices
        if total_frames <= n_frames:
            return list(range(start_idx, end_idx + 1))

        # Always include start and end
        indices = [start_idx]
        step = (end_idx - start_idx) / (n_frames - 1)

        for i in range(1, n_frames - 1):
            idx = int(round(start_idx + i * step))
            indices.append(idx)

        indices.append(end_idx)
        return indices
    

    def average_shot_embeddings(self, embedding_list):
        return np.mean(embedding_list, axis=0, keepdims=True)

    # ---- Retrieval ----
    def build_index(self, embeddings, video_metadata):
        d = embeddings.shape[1]  # dimension
        self.index = faiss.IndexFlatIP(d)  # cosine similarity if normalized
        self.index.add(embeddings)
        self.video_metadata = video_metadata

    def search(self, text_query, topk=5):
        text_embed = self.compute_text_embedding(text_query)
        text_embed = text_embed / np.linalg.norm(text_embed, axis=1, keepdims=True)  # normalize

        D, I = self.index.search(text_embed, topk)  # distances & indices
        results = [(self.video_metadata[i], float(D[0][j])) for j, i in enumerate(I[0])] # [(frame_path, sim_score),..]
        return results
