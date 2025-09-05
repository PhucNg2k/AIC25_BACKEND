import json

video_metadata = {}

FRAME_METADATA_FILE = "../Metadata/video_metadata_new.json"
with open(FRAME_METADATA_FILE, 'r') as f:
    video_metadata = json.load(f)

def get_metakey(video_name: str, frameIdx: int):
    return f"{video_name}_{frameIdx:06d}"


def get_frame_idx(meta_key):
    return video_metadata.get(meta_key, {}).get('frame_idx', None)

def get_pts_time(meta_key):
    return video_metadata.get(meta_key, {}).get('pts_time', None)

def get_video_fps(meta_key):
    return video_metadata.get(meta_key, {}).get('fps', 30)

def get_frame_path(meta_key):
    return video_metadata.get(meta_key, {}).get('frame_path', None)

def get_video_duration(meta_key):
    return video_metadata.get(meta_key, {}).get('video_duration', None)
