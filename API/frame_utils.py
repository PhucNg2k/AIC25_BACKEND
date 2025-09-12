import json, sys
import os
video_metadata = {}

API_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(API_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

print(ROOT_DIR)


FRAME_METADATA_FILE = "../Metadata/video_metadata_new.json"
FRAME_METADATA_FILE = "/home/phucuy2025/Documents/AIC_2025/VBS_system/AIC25_BACKEND/Metadata/video_metadata_new.json"
with open(os.path.abspath((FRAME_METADATA_FILE)), 'r') as f:
    video_metadata = json.load(f)

def get_metakey(video_name: str, frameIdx: int):
    return f"{video_name}_{frameIdx:06d}"

def get_frame_idx(meta_key):
    return video_metadata.get(meta_key, {}).get('frame_idx', None)

def get_pts_time(meta_key):
    return video_metadata.get(meta_key, {}).get('pts_time', None)

def get_video_fps(meta_key):
    return video_metadata.get(meta_key, {}).get('fps', None)

def get_frame_path(meta_key):
    return video_metadata.get(meta_key, {}).get('frame_path', None)

def get_video_duration(meta_key):
    return video_metadata.get(meta_key, {}).get('video_duration', None)
