
import os
import pandas as pd
import math
from typing import Optional, Dict, Any
from preprocess import process_string, splitter
from itertools import chain

def convert_row_to_document(row) -> Optional[Dict[str, Any]]:
    def _is_invalid(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip().lower() in ("", "none", "nan")
        if isinstance(value, float):
            return math.isnan(value)
        return False
    
    """Convert DataFrame row to OCR document"""
    try:
        # Handle different column layouts
        df_columns = row.index.tolist()
        if 'group' not in df_columns:
            doc = {
                "video_folder": str(row.get('video', '')),
                "video_name": str(row.get('image', '')),
                "frame_id": str(row.get('frame_id', '')),
                "text": str(row.get('text.1', ''))
            }
        else:
            doc = {
                "video_folder": str(row.get('group', '')),
                "video_name": str(row.get('video', '')),
                "frame_id": str(row.get('frame_id', '')),
                "text": str(row.get('text', ''))
            }
        
        # Validate required fields
        if any(_is_invalid(v) for v in [doc["video_folder"], doc["video_name"], doc["frame_id"]]):
            return None
            
        return doc
    except Exception as e:
            print(f"❌ Error converting row to document: {e}")
            return None

def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """ Process dataframe and return dataframe with chunked text, reserve format, column
    just chunk the text field
    """
    documents = []
    df_columns = df.columns
    
    print("DF columns: ", df_columns)
    
    for _, row in df.iterrows():
        doc = convert_row_to_document(row)
        if doc:
            text_ocr = doc['text']
            ocr_chunks = splitter.split_text(text_ocr)
            
            chunked_row = []            
            for chunk in ocr_chunks:
                tmp = {
                    "video_folder": doc['video_folder'],
                    "video_name": doc['video_name'],
                    "frame_id": doc['frame_id'],
                    "text": process_string(chunk)
                }
                chunked_row.append(tmp)
            
            documents.append(chunked_row)
    
    flattened_documents = list(chain.from_iterable(documents))

    new_df = pd.DataFrame(flattened_documents)
    return new_df    
        

def make_xl_file(df, save_dir, file_name):
    new_filename = f"{file_name}_chunked.xlsx"
    
    output_file = os.path.join(save_dir, new_filename)

    df.to_excel(output_file, index=False)
    print(f"Excel file saved as {output_file}")



DATA_FOLDER = os.path.abspath("../../REAL_DATA")
ocr_folders = ['ocr_b1', 'ocr_b2']

for ocr_folder in ocr_folders:
    folder_path = os.path.join(DATA_FOLDER, ocr_folder)
    
    save_dir = os.path.join(DATA_FOLDER, f"{ocr_folder}_chunked")
    os.makedirs(save_dir, exist_ok=True)
    
    if (not os.path.exists(folder_path)):
        print("\nSkipping: ", folder_path )
        continue
    
    for xl_file in sorted(os.listdir(folder_path)):
        file_name = os.path.splitext(xl_file)[0]
        filepath = os.path.join(folder_path, xl_file)
        print("\nProcessing: ", filepath)
        df = pd.read_excel(filepath)
        new_df = process_df(df)
        
        make_xl_file(new_df, save_dir, file_name)
        


