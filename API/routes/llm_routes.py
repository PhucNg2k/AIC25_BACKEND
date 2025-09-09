from fastapi import  HTTPException, APIRouter
from pydantic import BaseModel
from typing import List
import sys
import os


API_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.dirname(API_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    
from dependecies import GeminiClientDeps
from tools.translate_lang import translate_file
from config import QUERY_SOURCE
    
router = APIRouter(prefix="/llm", tags=["gemini llm"])


class TranslateResponse(BaseModel):
    file_name: str
    translated_text: List[str]
    message: str
    
class TranslateRequest(BaseModel):
    file_name: str


@router.post('/translate', response_model=TranslateResponse)
async def translate_query(payload: TranslateRequest, llm_client: GeminiClientDeps):
    # Validate file_name
    file_name = payload.file_name
    if not file_name or not isinstance(file_name, str):
        raise HTTPException(status_code=400, detail="Parameter 'file_name' is required and must be a string.")
    if any(sep in file_name for sep in ("..", "/", "\\")):
        raise HTTPException(status_code=400, detail="Invalid 'file_name'. Path traversal is not allowed.")

    # Ensure LLM client is available
    if llm_client is None or not hasattr(llm_client, "models"):
        raise HTTPException(status_code=503, detail="LLM service is not available or not configured.")

    # Build path and check existence
    fpath = os.path.join(QUERY_SOURCE, file_name)
    if not os.path.isfile(fpath):
        raise HTTPException(status_code=404, detail=f"File '{file_name}' not found in query source.")

    try:
        sentences = translate_file(fpath, llm_client)
        return TranslateResponse(file_name=file_name, translated_text=sentences, message="ok")
    except HTTPException:
        # Re-raise HTTPExceptions from downstream
        raise
    except Exception as e:
        # Detect likely API key / auth errors
        error_message = str(e)
        if any(token in error_message.lower() for token in ["unauthorized", "forbidden", "api key", "invalid key", "401", "403"]):
            raise HTTPException(status_code=502, detail="LLM service authentication failed. Check API key configuration.")
        raise HTTPException(status_code=500, detail=f"Error translating: {error_message}")
        
    
# will implement route for agentic AI