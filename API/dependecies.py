from typing import Annotated, Any
from fastapi import Depends, HTTPException
from retrieve_vitL import index, metadata

import os, sys
from ElasticSearch.ESclient import (
    OCRClient,
    ASRClient
)

from google import genai

# Parent directory to import retrieve module, make script-friendly
API_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(API_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from config import OCR_INDEX_NAME, ASR_INDEX_NAME

async def get_search_resources():
    """Dependency to provide search resources"""
    try:
        if index is None or metadata is None:
            raise HTTPException(status_code=500, detail="Search index not loaded")
        
        yield {
            "index": index,
            "metadata": metadata,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
# dependecies.py
def get_es_config():
    # swap this body to read from AWS SM, Vault, etc.
    return {
        "hosts": [os.getenv("ES_LOCAL_URL")],
        "api_key": os.getenv("ES_LOCAL_API_KEY"),
    }

async def get_es_client():
    try:
        cfg = get_es_config()
        if not cfg["hosts"][0] or not cfg["api_key"]:
            raise HTTPException(status_code=500, detail="Missing ES config")
        yield cfg
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Elasticsearch connection error: {str(e)}")

async def get_ocr_client(es_connection: Annotated[dict, Depends(get_es_client)]):
    """Dependency to provide OCR client"""
    try:
        ocr_client = OCRClient(
            hosts=es_connection["hosts"], 
            api_key=es_connection["api_key"], 
            index_name=OCR_INDEX_NAME
        )
        
        # Test the connection
        try:
            ocr_client.es.info()
        except Exception:
            raise HTTPException(status_code=500, detail="Cannot connect to Elasticsearch")
        
        yield ocr_client
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR client error: {str(e)}")

async def get_asr_client(es_connection: Annotated[dict, Depends(get_es_client)]):
    """Dependency to provide ASR client"""
    try:
        asr_client = ASRClient(
            hosts=es_connection["hosts"], 
            api_key=es_connection["api_key"], 
            index_name=ASR_INDEX_NAME
        )
        
        # Test the connection
        try:
            asr_client.es.info()
        except Exception:
            raise HTTPException(status_code=500, detail="Cannot connect to Elasticsearch")
        
        yield asr_client
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR client error: {str(e)}")


async def get_llm_client():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    yield client




search_resource_Deps = Annotated[dict, Depends(get_search_resources)]

OCRClientDeps = Annotated[OCRClient, Depends(get_ocr_client)]
ASRClientDeps = Annotated[ASRClient, Depends(get_asr_client)]
GeminiClientDeps = Annotated[genai.Client, Depends(get_llm_client)]