from typing import Tuple, List
from models.response import ImageResult

def convert_ImageList(raw_results):
    if len(raw_results) <= 0:
        return []
    
    results = []
    for raw_result in raw_results:
        result = ImageResult(**raw_result)
        results.append(result)
    return results

# normalize to (0,1) range
def normalize_score(frame_list: List[ImageResult]) -> List[ImageResult]:
    if not frame_list:
        return []

    scores = [frame['score'] for frame in frame_list]
    min_score = min(scores)
    max_score = max(scores)

    # Avoid division by zero if all scores are the same
    if min_score == max_score:
        for frame in frame_list:
            frame['score'] = 0.5
    else:
        for frame in frame_list:
            frame['score'] = (frame['score'] - min_score) / (max_score - min_score)

    return frame_list


def sort_score_results(results: List[ImageResult], reverse=True) -> List[ImageResult]:
    if not results:
        return []
    return sorted(results, key=lambda r: r['score'], reverse=reverse)


def get_weighted_results_quantity(frame_list: List[ImageResult], weight_score: float) -> List[ImageResult]:
    if weight_score < 0:
        return []
    
    frame_list = sort_score_results(frame_list, reverse=True)
    count = max(1, min(len(frame_list), int(len(frame_list) * weight_score)))
    # Return the top portion of the list
    return frame_list[:count]

def get_weighted_results_fuse(frame_list: List[ImageResult], weight_score: float) -> List[ImageResult]:
    for f in frame_list:
        f["score"] *= weight_score
        
    return frame_list

def get_weighted_union_results(list_results: List[List[ImageResult]], weight_list: List[float], fuse=True) ->  List[List[ImageResult]]:
    if ( len(list_results) != len (weight_list) ):
        return []
    
    weighted_modality_results = []

    for results, weight in zip(list_results, weight_list):
        if weight <= 0:
            continue  # Skip if weight is zero or negative
        
        if fuse:
            weighted_results = get_weighted_results_fuse(results, weight)
        else:
            weighted_results = get_weighted_results_quantity(results, weight)

        # Keep each modality separate
        weighted_modality_results.append(weighted_results)

    return weighted_modality_results




def apply_bonus_obj():
    pass



def get_intersection_results(list_results: List[List[ImageResult]]) -> List[ImageResult]:
    """
    Return ImageResults that appear in EVERY list based on (video_name, frame_idx).
    No sorting or score aggregation here. If an ImageResult appears in multiple
    lists, we return the instance from the FIRST list to preserve a stable source.
    """
    if not list_results:
        return []
    if any((lst is None or len(lst) == 0) for lst in list_results):
        return []

    if len(list_results) == 1:
        return list_results[0]

    def make_key(item: ImageResult):
        return (item['video_name'], item['frame_idx'])

    # Compute intersection of keys across all lists
    key_sets = [{make_key(item) for item in lst} for lst in list_results]
    intersect_keys = set.intersection(*key_sets)
    if not intersect_keys:
        return []

    # Preserve order from the first list: pick items whose key is in intersection
    first_list = list_results[0]
    intersection_results: List[ImageResult] = []
    for item in first_list:
        if make_key(item) in intersect_keys:
            intersection_results.append(item)

    return intersection_results

def parse_frame_file(frame_data: str) -> Tuple[str, str]:
    """
    Parse a frame identifier string into (video_name, frame_index).

    Expected formats:
    - "<video_name>_<frame_index>"
      where <video_name> may contain underscores (e.g., "L30_V001_0030").
      We split on the last underscore.
    - Optional extension is ignored (e.g., "L30_V001_0030.jpg").

    Returns:
        (video_name, frame_index_as_string)

    Raises:
        ValueError: if the input cannot be parsed.
    """
    if frame_data is None:
        raise ValueError("frame_data cannot be None")

    token = frame_data.strip()
    if not token:
        raise ValueError("frame_data cannot be empty")

    # Trim extension if present
    if "." in token:
        token = token.rsplit(".", 1)[0]

    # Split on last underscore
    if "_" not in token:
        raise ValueError(f"Invalid frame identifier: '{frame_data}'")

    video_name, frame_idx = token.rsplit("_", 1)

    if not video_name or not frame_idx:
        raise ValueError(f"Invalid frame identifier: '{frame_data}'")

    # Ensure frame_idx is numeric (but return as string to let caller cast)
    try:
        int(frame_idx)
    except Exception:
        raise ValueError(f"Frame index is not an integer in '{frame_data}'")

    return video_name, frame_idx