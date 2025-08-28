import os
import cv2
import pickle
from pathlib import Path
import json

video_dir = "../REAL_DATA/Data/video"
output_dir = "./Metadata"
metadata_file = os.path.join(output_dir, "video_metadata.pkl")
metadata_file_json = os.path.join(output_dir, "metadata.json")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get list of video files and sort them numerically
video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

# Sort videos numerically by their number (Video_0001, Video_0002, etc.)
def extract_video_number(filename):
    # Extract the number from "Video_0001.mp4" -> 1
    try:
        # Remove extension and split by underscore
        name_without_ext = os.path.splitext(filename)[0]
        number_part = name_without_ext.split('_')[-1]
        return int(number_part)
    except (ValueError, IndexError):
        return 0  # Default value for files that don't match pattern

def extract_video_metadata(video_path):
    """Extract metadata from a video file using OpenCV"""
    try:
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        # Extract metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate duration in seconds
        duration = frame_count / fps if fps > 0 else 0
        
        # Release video capture
        cap.release()
        
        metadata = {
            'fps': fps,
            'width': width,
            'height': height,
            'resolution': f"{width}x{height}",
            'frame_count': frame_count,
            'duration_seconds': duration,
            'duration_formatted': format_duration(duration),
            'video_path': video_path
        }
        
        return metadata
        
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return None

def format_duration(seconds):
    """Format duration from seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

# Sort videos
video_files.sort(key=extract_video_number)

print(f"Found {len(video_files)} videos. Processing...")
print("=" * 60)

# Dictionary to store all video metadata
video_metadata = {}

# Process each video
for i, video_file in enumerate(video_files):
    video_path = os.path.join(video_dir, video_file)
    video_name = os.path.splitext(video_file)[0]  # Remove extension for key
    
    print(f"Processing {i+1}/{len(video_files)}: {video_file}")
    
    # Extract metadata
    metadata = extract_video_metadata(video_path)
    
    if metadata:
        video_metadata[video_name] = metadata
        
        # Print summary
        print(f"  ✓ FPS: {metadata['fps']:.2f}")
        print(f"  ✓ Resolution: {metadata['resolution']}")
        print(f"  ✓ Duration: {metadata['duration_formatted']} ({metadata['frame_count']} frames)")
    else:
        print(f"  ✗ Failed to process {video_file}")
    
    print("-" * 40)

# Save metadata to pickle file
try:
    with open(metadata_file, 'wb') as f:
        pickle.dump(video_metadata, f)
    
    with open(metadata_file_json, 'w') as f:
        json.dump(video_metadata, f)
    
    print(f"\n✅ Successfully saved metadata for {len(video_metadata)} videos to: {metadata_file} & {metadata_file_json}")
    
    # Print summary statistics
    if video_metadata:
        print("\n📊 Summary Statistics:")
        print("=" * 40)
        
        total_duration = sum(meta['duration_seconds'] for meta in video_metadata.values())

        total_frames = sum(meta['frame_count'] for meta in video_metadata.values())
        
        avg_fps = sum(meta['fps'] for meta in video_metadata.values()) / len(video_metadata)
        unique_resolutions = set(meta['resolution'] for meta in video_metadata.values())
        
        print(f"Total videos processed: {len(video_metadata)}")
        print(f"Total duration: {format_duration(total_duration)}")
        print(f"Total frames: {total_frames:,}")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Unique resolutions: {', '.join(sorted(unique_resolutions))}")
        
        # Show sample metadata for first video
        if video_metadata:
            first_video = next(iter(video_metadata.items()))
            print(f"\n📋 Sample metadata structure (from {first_video[0]}):")
            for key, value in first_video[1].items():
                print(f"  {key}: {value}")
    
except Exception as e:
    print(f"❌ Error saving metadata: {str(e)}")

print(f"\n🎬 Video metadata extraction complete!")