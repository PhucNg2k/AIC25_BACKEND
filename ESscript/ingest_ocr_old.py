from dotenv import load_dotenv
import os
import pandas as pd

from elasticsearch.exceptions import RequestError
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

load_dotenv()
es_url = os.getenv("ES_LOCAL_URL")
es_api_key = os.getenv("ES_LOCAL_API_KEY")

from API.ElasticSearch.ESclient import OCRClient

ocr_client = OCRClient(
            hosts=[es_url], 
            api_key=es_api_key, 
            index_name='ocr_index_chunked'
        )

ocr_client.create_index(force=True)

target_folder = ['ocr_b1_chunked', 'ocr_b2_chunked']
for folder in target_folder:
    DATA_PATH = f"../../REAL_DATA/{folder}"
    
    if (not os.path.exists(DATA_PATH)):
        print("\nSkipping: ", DATA_PATH )
        continue
    print("\nProcessing: ", folder)
    for xlfile in sorted(os.listdir(DATA_PATH)):
        filepath = os.path.join(DATA_PATH, xlfile)
        print("\nProcessing: ", filepath)
        df = pd.read_excel(filepath)
        ocr_client.bulk_index_from_dataframe(df)
        print("Ingested: ", filepath)


    print("Ingesting OCR done")