import sys
import os
import numpy as np
import torch
import cv2
sys.path.append(os.path.join(os.path.dirname(__file__), 'inference-pytorch'))

from CustomTransNet import MyTransNet

weight_path = "./inference-pytorch/transnetv2-pytorch-weights.pth"
video_dir = "../REAL_DATA/Data/video"
output_dir = "./ExtractedFrames"
scenetxt_dir = "./ExtractedScene"

def inference_video(video_path: str):
    with torch.no_grad():
        frames_np = model.get_video_frames(video_path)

        input_video = torch.from_numpy(frames_np).unsqueeze(0)

        single_frame_pred, all_frame_pred = model(input_video.cuda())
        single_frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()
        all_frame_pred = torch.sigmoid(all_frame_pred["many_hot"]).cpu().numpy()
        return single_frame_pred, all_frame_pred

def get_scenes_from_video(video_path: str):
    video_filename = os.path.basename(video_path)
    single_frame_pred, all_frame_pred = inference_video(video_path)
    scenes = model.predictions_to_scenes(single_frame_pred.squeeze())
    
    # Save scenes to file
    scenes_filename = os.path.join(scenetxt_dir, f"{video_filename}.scenes.txt")
    np.savetxt(scenes_filename, scenes, fmt="%d")
    
    return scenes

def get_inter_frames_idx(n_frames: int, bounds: tuple):
    '''Divide index equally into n_frames within bounds, inclusive'''
    start_idx, end_idx = bounds
    
    if n_frames == 1:
        return [start_idx]
    elif n_frames == 2:
        return [start_idx, end_idx]
    else:
        # For 3 or more frames, include start, middle, and end
        step = (end_idx - start_idx) / (n_frames - 1)
        frames_idx = []
        for i in range(n_frames):
            frame_idx = int(start_idx + i * step)
            frames_idx.append(frame_idx)
        return frames_idx

def capture_multiple_frames(video_path: str, frame_indexes: list, output_dir: str, video_name: str):
    '''extract multiple frames from a video by seeking to specific positions'''
    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        extracted_frames = []
        
        print(f"    Extracting {len(frame_indexes)} frames...")
        
        for frame_idx in frame_indexes:
            # Seek to specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Create output filename
                frame_filename = f"{video_name}_frame_{frame_idx:06d}.jpg"
                frame_output_path = os.path.join(output_dir, frame_filename)
                
                # Save the frame
                cv2.imwrite(frame_output_path, frame)
                extracted_frames.append(frame_output_path)
                print(f"      Frame {frame_idx} saved to {frame_filename}")
            else:
                print(f"      Warning: Could not read frame {frame_idx}")
        
        cap.release()
        return extracted_frames
        
    except Exception as e:
        print(f"Error extracting frames from {video_path}: {e}")
        if 'cap' in locals():
            cap.release()
        return []

def main():
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(scenetxt_dir, exist_ok=True)
    
    FRAME_SAMPLES = 3
    
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
    
    video_files.sort(key=extract_video_number)
    
    print(f"Found {len(video_files)} videos. Processing order:")
    for i, video in enumerate(video_files):
        print(f"  {i+1}. {video}")
    
    for video_filename in video_files:
        video_path = os.path.join(video_dir, video_filename)
        print(f"Processing: {video_filename}")
        
        try:
            # Get scenes from video
            scenes = get_scenes_from_video(video_path)
            
            # Create subdirectory for this video's frames
            video_name = os.path.splitext(video_filename)[0]
            video_frames_dir = os.path.join(output_dir, video_name)
            os.makedirs(video_frames_dir, exist_ok=True)
            
            print(f"Found {len(scenes)} scenes")
            
            # Collect all frame indexes we need from all scenes
            all_frame_indexes = []
            for scene_idx, scene in enumerate(scenes):
                print(f"  Processing scene {scene_idx + 1}/{len(scenes)}: frames {scene[0]}-{scene[1]}")
                
                # Get target frames (start, middle, end)
                target_frames = get_inter_frames_idx(FRAME_SAMPLES, scene)
                all_frame_indexes.extend(target_frames)
            
            # Extract all frames in one pass
            if all_frame_indexes:
                print(f"  Extracting {len(all_frame_indexes)} total frames...")
                extracted_frames = capture_multiple_frames(video_path, all_frame_indexes, video_frames_dir, video_name)
                print(f"  Successfully extracted {len(extracted_frames)} frames")
                        
        except Exception as e:
            print(f"Error processing {video_filename}: {e}")
            continue

        break

if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        device = torch.device('cuda')
    else:
        print("CUDA not available. Using CPU.")
        device = torch.device('cpu')
    
    # Initialize model
    model = MyTransNet(weight_path)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    print(f"Processing videos from: {video_dir}")
    print(f"Output frames will be saved to: {output_dir}")
    
    main()
    
    print("Frame extraction completed!")
