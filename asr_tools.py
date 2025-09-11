import os
import json
import re
import unicodedata
from typing import List, Optional, Dict

from API.frame_utils import get_video_fps, get_metakey
from API.group_utils import get_group_frames
from nltk.util import ngrams

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def _normalize_text(text: str) -> str:
    if text is None:
        return ""
    # lowercase + strip diacritics + collapse whitespace
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    norm = _normalize_text(text)
    if not norm:
        return []
    return norm.split(" ")


def find_closest_keyframe(group_keyframes: List[Dict], target_frame_id: int) -> Optional[Dict]:
    if not group_keyframes:
        return None
    closest = None
    best_dist = None
    
    for frame in group_keyframes:
        frame_id = int(frame.get('frame_idx', -1))
        if frame_id < 0:
            continue
        
        dist = abs(frame_id - target_frame_id)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            closest = frame
            
    print('CLOSEST FRAME: ', closest)
    return closest or group_keyframes[0]

def matching_ngram(data: List[Dict], text_content: str, query_text: Optional[str] = None) -> Optional[float]:
    """Rank transcript snippets by relevance to the retrieved chunk text and optional query,
    preferring longer phrase matches (7->3 grams) and skipping very short snippets.

    Returns best snippet midpoint time (seconds) or None.
    """
    hit_text_norm = _normalize_text(text_content)
    if not hit_text_norm:
        return None

    query_norm = _normalize_text(query_text) if query_text else ""
    query_tokens_list = [t for t in (query_norm.split(" ") if query_norm else []) if t]
    # Dynamic n-gram window based on query length, with special handling for very short queries
    if not query_tokens_list:
        n_values = [7, 6, 5, 4, 3]
        query_tokens = set()
    else:
        q_len = len(query_tokens_list)
        if q_len < 3:
            # For very short queries, use safe fixed range and ignore query overlap
            n_values = [5, 4, 3]
            query_tokens = set()
        else:
            n_max = max(3, min(10, q_len))
            n_values = list(range(n_max, 2, -1))  # n_max .. 3
            query_tokens = set(query_tokens_list)

    best = None  # (score_tuple, mid_time)

    for snippet in data:
        snippet_text = snippet.get('text', '')
        start_time = snippet.get('start', None)
        duration = snippet.get('duration', 0.0)
        if start_time is None:
            continue

        tokens = _tokenize(snippet_text)
        if len(tokens) < 3:
            continue  # ignore too-short snippets

        # compute longest n-gram match length in hit text (prefer longer)
        longest_hit_ngram = 0
        match_pos = None
        # search descending n sizes for efficiency
        for n in n_values:
            if len(tokens) < n:
                continue
            found = False
            for gram_tokens in ngrams(tokens, n):
                gram = " ".join(gram_tokens)
                pos = hit_text_norm.find(gram)
                if pos != -1:
                    longest_hit_ngram = n
                    match_pos = pos
                    found = True
                    break  # take first occurrence for this n
            if found:
                break

        if longest_hit_ngram == 0:
            # try middle split as a weaker signal
            mid = len(tokens) // 2
            left = " ".join(tokens[:mid]).strip()
            right = " ".join(tokens[mid:]).strip()
            split_match = 0
            pos_cand = None
            if left and left in hit_text_norm:
                split_match = max(split_match, len(left.split(" ")))
                pos_cand = hit_text_norm.find(left)
            if right and right in hit_text_norm:
                split_match = max(split_match, len(right.split(" ")))
                pos_cand = hit_text_norm.find(right) if pos_cand is None else pos_cand
            longest_hit_ngram = split_match
            match_pos = pos_cand

        if longest_hit_ngram == 0:
            continue  # no evidence of presence in hit text

        # query similarity (token overlap)
        query_overlap = 0.0
        if query_tokens:
            s_tokens = set(tokens)
            inter = len(s_tokens & query_tokens)
            union = len(s_tokens | query_tokens) or 1
            query_overlap = inter / union

        # snippet midpoint time
        #mid_time = float(start_time) + float(duration) / 2.0
        mid_time = float(start_time)

        # score tuple: higher is better
        score = (
            longest_hit_ngram,     # primary: longest phrase present in hit
            query_overlap,          # secondary: closeness to query
            - (match_pos or 0),     # tertiary: earlier occurrence (smaller pos -> larger score)
            len(tokens),            # quaternary: longer snippet (more specific)
        )

        if best is None or score > best[0]:
            best = (score, mid_time)

    return None if best is None else best[1]


def predict_from_timeline(time_line_snippet, text_query):
    pass


def jaccard_similarity(set1, set2):
    # intersection of two sets
    intersection = len(set1.intersection(set2))
    # Unions of two sets
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents='unicode',
    token_pattern=r'\b\w+\b',  # word boundaries
    max_features=1000,
    stop_words=None  # keep all words for Vietnamese
)

def estimate_keyframes_tf_idf(lookup_data, retrieved_chunk_txt, query_text):
    """Use TF-IDF cosine similarity + query keyword matching to find best snippet."""
    if not lookup_data or not retrieved_chunk_txt:
        return None
    
    corpus = [snippet.get('text', '') for snippet in lookup_data]
    
    # Fit TF-IDF on corpus
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    
    # Transform chunk to same space
    chunk_vector = tfidf_vectorizer.transform([retrieved_chunk_txt])
    
    # Compute TF-IDF similarities
    tfidf_similarities = cosine_similarity(chunk_vector, tfidf_matrix).flatten()
    
    # Query keyword matching scores
    query_scores = []
    if query_text:
        query_norm = _normalize_text(query_text)
        query_tokens = set(query_norm.split(" ")) if query_norm else set()
        
        for snippet in lookup_data:
            snippet_text = snippet.get('text', '')
            snippet_norm = _normalize_text(snippet_text)
            snippet_tokens = set(snippet_norm.split(" ")) if snippet_norm else set()
            
            # Jaccard similarity with query
            if query_tokens and snippet_tokens:
                query_score = jaccard_similarity(query_tokens, snippet_tokens)
            else:
                query_score = 0.0
            query_scores.append(query_score)
    else:
        query_scores = [0.0] * len(lookup_data)
    
    # Combined scoring: TF-IDF similarity + query alignment
    # Weight: 70% TF-IDF, 30% query matching
    combined_scores = []
    for i, snippet in enumerate(lookup_data):
        tfidf_score = tfidf_similarities[i]
        query_score = query_scores[i]
        combined = 0.7 * tfidf_score + 0.3 * query_score
        combined_scores.append(combined)
    
    # Find best match
    best_idx = max(range(len(combined_scores)), key=lambda i: combined_scores[i])
    best_snippet = lookup_data[best_idx]
    
    # Return midpoint time
    start_time = best_snippet.get('start', 0)
    duration = best_snippet.get('duration', 0)
    return float(start_time)


def get_estimate_keyframes(orginal_text: str, video_name: str, query_text: Optional[str] = None) -> Dict:
    if video_name[0] == "L":
        batch_name = 'b1'
    else: 
        batch_name = 'b2'
        
    # Build absolute path: [workspace_root]/REAL_DATA/asr_data_combined/asr_{batch_name}/raw/{video_name}.json
    backend_dir = os.path.dirname(os.path.abspath(__file__))          # .../AIC25_BACKEND
    workspace_root = os.path.dirname(backend_dir)                      # .../Competition/AIC25
    raw_dir = os.path.join(workspace_root, 'REAL_DATA', 'asr_data_combined', f'asr_{batch_name}', 'raw')
    lookup_file = os.path.join(raw_dir, f'{video_name}.json')

    try:
        with open(lookup_file, 'r') as f:
            data = json.load(f)
    except Exception:
        data = []
    
    group_keyframes = get_group_frames(video_name)
    if not group_keyframes:
        # Minimal fallback
        return {
            'video_name': video_name,
            'frame_idx': -1,
            'image_path': "/",
            'score': 0.0,
            'pts_time': -1.0,
        }
    vid_fps = None
    for frame in group_keyframes:
        video = frame['video_name']    
        frame_idx = frame['frame_idx']
        metakey = get_metakey(video, frame_idx)
        vid_fps = get_video_fps(metakey)
        if vid_fps: # ensure get vid_fps from available keyframes
            break
    if not vid_fps:
        return group_keyframes[0]

    #anchor_time = matching_ngram(data, orginal_text, query_text) # !!
    anchor_time = estimate_keyframes_tf_idf(data, orginal_text, query_text)
    if anchor_time is None:
        return group_keyframes[0]
    
    target_frame_id = round(anchor_time * vid_fps)
    closest_keyframe = find_closest_keyframe(group_keyframes, target_frame_id)
    
    print('\n', orginal_text)
    print("ANCHOR TIME: ", anchor_time)
    print("TARGET FRAME ID: ", target_frame_id)

    return closest_keyframe or group_keyframes[0]