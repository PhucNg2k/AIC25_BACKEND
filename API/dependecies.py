from typing import Annotated, Any
from fastapi import Depends, HTTPException
from retrieve_vitL import index, metadata
from elasticsearch import Elasticsearch
import os
from dotenv import load_dotenv
from ElasticSearch.ESclient import ESClientBase, OCRClient, ASRClient


load_dotenv()


OCR_INDEX_NAME = 'ocr_index'
ASR_INDEX_NAME = 'asr_index'

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
    

async def get_es_client():
    """Dependency to provide Elasticsearch client"""
    try:
        es_url = os.getenv("ES_LOCAL_URL")
        es_api_key = os.getenv("ES_LOCAL_API_KEY")
        
        if not es_url:
            raise HTTPException(status_code=500, detail="ES_LOCAL_URL environment variable not set")
        if not es_api_key:
            raise HTTPException(status_code=500, detail="ES_LOCAL_API_KEY environment variable not set")
        
        # Return connection info instead of client instance
        yield {
            "hosts": [es_url],
            "api_key": es_api_key
        }
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

search_resource_Deps = Annotated[dict, Depends(get_search_resources)]

OCRClientDeps = Annotated[OCRClient, Depends(get_ocr_client)]
ASRClientDeps = Annotated[ASRClient, Depends(get_asr_client)]