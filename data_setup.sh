#!/bin/bash

# Script to consolidate all .mp4 files from nested video folders into a central location
# This will move files from Data/Videos_*/video/*.mp4 to Data/video/*.mp4

# Set the base directory
<<<<<<< HEAD
BASE_DIR="../REAL_DATA/Data"
VIDEOS_DIR="$BASE_DIR/video"
=======
BASE_DIR="../REAL_DATA/Data/video"
VIDEOS_DIR="/media/phucuy2025/KingstonUSB/b2_videos"
>>>>>>> 958a4cc (testing)

# Create the central video directory if it doesn't exist
echo "Creating central video directory: $VIDEOS_DIR"
mkdir -p "$BASE_DIR"

# Counter for moved files
moved_count=0
skipped_count=0

echo "Starting to consolidate video files..."

# Find all Videos_* directories
for videos_folder in "$VIDEOS_DIR"/Videos_*; do
    if [ -d "$videos_folder" ]; then
        echo "Processing: $videos_folder"
        
        # Check if there's a video subfolder
        if [ -d "$videos_folder/video" ]; then
            # Process each .mp4 file in the video subfolder
            for video_file in "$videos_folder/video"/*.mp4; do
                if [ -f "$video_file" ]; then
                    filename=$(basename "$video_file")
                    target_path="$VIDEOS_DIR/$filename"
                    
                    # Check if file already exists in target
                    if [ -f "$target_path" ]; then
                        echo "  Skipping $filename (already exists in target)"
                        ((skipped_count++))
                    else
                        echo "  Moving $filename"
                        mv "$video_file" "$target_path"
                        ((moved_count++))
                    fi
                fi
            done
        else
            echo "  No video subfolder found in $videos_folder"
        fi
    fi
done

echo ""
echo "Consolidation complete!"
echo "Files moved: $moved_count"
echo "Files skipped (already existed): $skipped_count"
echo "Total files in central video folder: $(find "$VIDEOS_DIR" -name "*.mp4" | wc -l | tr -d ' ')"
echo ""
echo "Central video folder location: $VIDEOS_DIR"
