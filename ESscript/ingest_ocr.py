from dotenv import load_dotenv
import os
import pandas as pd

from elasticsearch.exceptions import RequestError
from getpass import getpass
import sys

load_dotenv()
es_url = os.getenv("ES_LOCAL_URL")
es_api_key = os.getenv("ES_LOCAL_API_KEY")


API_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(API_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from Code.API.ElasticSearch.ESclient import OCRClient

DATA_PATH = "../REAL_DATA/ocr_b1"

ocr_client = OCRClient(
            hosts=[es_url], 
            api_key=es_api_key, 
            index_name='ocr_index'
        )

ocr_client.create_index(force=True)


for xlfile in sorted(os.listdir(DATA_PATH)):
    filepath = os.path.join(DATA_PATH, xlfile)
    print("\nProcessing: ", filepath)
    df = pd.read_excel(filepath)
    ocr_client.bulk_index_from_dataframe(df)
    print("Ingested: ", filepath)


print("Ingesting OCR done")