import os
from pytubefix import YouTube
from pytubefix.cli import on_progress

yt_links = []
videos_dir = "Videos"


def download_url(url: str, out_dir: str, high_res : bool = True):
    yt = YouTube(url, on_progress_callback=on_progress)
    if high_res:
        ys = yt.streams.get_highest_resolution()
    else:
        ys = yt.streams.get_lowest_resolution()
    
    # Download the video
    file_path = ys.download(output_path=out_dir)
    
    # Get the filename without extension
    filename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(filename)[0]
    file_ext = os.path.splitext(filename)[1]
    
    return file_path, name_without_ext, file_ext

def main():
    with open("yt_links.txt") as f:
        for lines in f.readlines():
            link = lines.strip()
            yt_links.append(link )

    for idx, url in enumerate(yt_links):
        # Download the video
        print(f"Downloading {url}...")
        file_path, original_name, file_ext = download_url(url, videos_dir)
        
        # Create new filename with sequential numbering
        new_filename = f"Video_{idx+1:04d}{file_ext}"
        new_file_path = os.path.join(videos_dir, new_filename)
        
        # Rename the downloaded file
        if os.path.exists(file_path):
            os.rename(file_path, new_file_path)
            print(f"Downloaded and renamed: {original_name} -> {new_filename}\n")
        

if __name__ == "__main__":
    os.makedirs(videos_dir, exist_ok=True)
    main()