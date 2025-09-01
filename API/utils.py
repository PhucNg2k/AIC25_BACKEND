from typing import Tuple
from models import ImageResult

def convert_ImageList(raw_results):
    results = []
    for raw_result in raw_results:
        result = ImageResult(**raw_result)
        results.append(result)
    return results

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