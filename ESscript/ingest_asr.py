from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import sys
import json
from typing import List, Dict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from API.ElasticSearch.ESclient import ASRClient


def main():
    load_dotenv()
    es_url = os.getenv("ES_LOCAL_URL")
    es_api_key = os.getenv("ES_LOCAL_API_KEY")
    index_name = os.getenv("ASR_INDEX_NAME", "asr_index_chunked")

    client = ASRClient(
        hosts=[es_url],
        api_key=es_api_key,
        index_name=index_name
    )

    client.create_index(force=True)

    target_folder = ['chunked_asr']
    for folder in target_folder:
        DATA_PATH = f"../../REAL_DATA/{folder}"
        
        if (not os.path.exists(DATA_PATH)):
            print("\nSkipping: ", DATA_PATH )
            continue
        
        print("\nProcessing: ", folder)
        for csv_file in sorted(os.listdir(DATA_PATH)):
            filepath = os.path.join(DATA_PATH, csv_file)
            print("\nProcessing: ", filepath)
            df = pd.read_csv(filepath)
            client.bulk_index_from_dataframe(df)
            print("Ingested: ", filepath)


if __name__ == "__main__":
    main()