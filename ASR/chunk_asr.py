import os
import pandas as pd 
from asr_preprocess import splitter

SAVE_DIR = "../../REAL_DATA/chunked_asr"

# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

BASE_DIR = "../../REAL_DATA/asr_data"
target_dir = ['asr_b1', 'asr_b2']
chunk_counted = 0

for dir in target_dir:
    full_folder = os.path.join(BASE_DIR, dir, 'full')

    for txt_file in sorted(os.listdir(full_folder)):
        video_name = os.path.splitext(txt_file)[0]

        csv_file = f'{video_name}.csv'

        with open(os.path.join(full_folder, txt_file), 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        chunks = splitter.split_text(text_content)

        df_data = []

        for chunk in chunks:
            data_row = {
                "video_name": video_name,
                "text": chunk.strip()
            }
            df_data.append(data_row)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(df_data)
        csv_path = os.path.join(SAVE_DIR, csv_file)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Saved {len(df)} chunks to {csv_path}")
        
        chunk_counted += len(df)


print('CHUNK COUNTED: ', chunk_counted)