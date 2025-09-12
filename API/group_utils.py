import json
from frame_utils import get_metakey, get_pts_time, get_frame_path
import os

group_metadata = {}

GROUP_METADATA_FILE = "../Metadata/grouped_keyframes_metadata.json"
with open(GROUP_METADATA_FILE, 'r') as f:
    group_metadata = json.load(f)
    

def get_group_frames(video_name: str):
    video_name = video_name.upper()
    
    final_res = []
    frame_ls = []
    
    found = False
    for pfolder, videos in group_metadata.items():
        # pfolder: video_name folder, videos: each leaf folder
        for key_name, frames in videos.items():
            if video_name == key_name: # L21_V001
                frame_ls = frames # list of keyframe (.webp)
                found = True
                break
        if found:
            break

    if not frame_ls:
        return []

    for i, frame_f in enumerate(frame_ls):
        base = os.path.splitext(frame_f)[0]  # 'f000648'
        frame_idx = int(base[1:])  # remove the first 'f' and cast to int
    
        metakey = get_metakey(video_name, frame_idx)
        pts_time = get_pts_time(metakey)
        image_path = get_frame_path(metakey)

        score = (len(frame_ls)-i) / 10

        result = {
            'video_name': video_name,
            'frame_idx': frame_idx,
            'image_path': image_path,
            'score': score,
            'pts_time': pts_time
        }
        
        final_res.append(result)
    
    return final_res

