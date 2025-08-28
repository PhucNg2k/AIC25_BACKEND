from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse


from utils import parse_frame_file

from pydantic import BaseModel
from typing import List, Optional, Annotated
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieve import search_query, ImageResult, index, metadata

'''
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
translate_model_name = "VietAI/envit5-translation"
translate_tokenizer = AutoTokenizer.from_pretrained(translate_model_name)  
translate_model = AutoModelForSeq2SeqLM.from_pretrained(translate_model_name)
translate_model = translate_model.to(device)


async def translate_query(vi_query: str):
    query_string = f"vi: {vi_query}"
    outputs = translate_model.generate(translate_tokenizer(query_string, return_tensors="pt", padding=True).input_ids.to(device), max_length=512)
    result = translate_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    result = result[3:].strip()
    return result
'''
# Request/Response Models
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10

class SearchResponse(BaseModel):
    success: bool
    query: str
    results: List[ImageResult]
    total_results: int
    message: Optional[str] = None


app = FastAPI(title="Text-to-Image Retrieval API", version="1.0.0")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def fake_video_streamer():
    for i in range(10):
        yield b"some fake video bytes"

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Text-to-Image Retrieval API is running", "status": "healthy"}


@app.get("/get_frames", response_class=FileResponse)
async def get_frame(frame_name: Annotated[str, Query()]):
    frame_folder = "../ExtractedFrames/"
    video_name, frame_idx = parse_frame_file(frame_name)
    file_path = os.path.join(frame_folder, video_name, f"{frame_name}.jpg")
    return file_path

@app.get("/get_videos")
def get_videos(video_name: str):
    video_folder = "../Videos"
    video_path = os.path.join(video_folder, f"{video_name}.mp4")
    
    def iter():
        with open(video_path, 'rb') as video: # read binary
            yield from video

    return StreamingResponse(iter(), media_type="video/mp4")

@app.post("/textSearch", response_model=SearchResponse)
async def text_search(request: SearchRequest):
    """
    Search for images based on text query
    
    Args:
        request: SearchRequest containing query string and optional top_k parameter
        
    Returns:
        SearchResponse with search results
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty")
    
    if request.top_k <= 0 or request.top_k > 100:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 100")
    
    try:
        # Check if index and metadata are loaded
        if index is None or metadata is None:
            raise HTTPException(
                status_code=500, 
                detail="Search index not loaded. Please ensure indexing has been completed."
            )
        
        # Search for similar images
    
        # en_query = translate_query(request.query.strip())
        en_query = request.query.strip()
        print("SEARCH QUERY: ", en_query)
        results = search_query(en_query.strip().lower(), index, metadata, top_k=request.top_k)
        
        return SearchResponse(
            success=True,
            query=request.query.strip(),
            results=results,
            total_results=len(results),
            message=f"Found {len(results)} results for query: '{request.query.strip()}'"
        )
        
    except Exception as e:
        print(f"Error during search: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during search: {str(e)}"
        )
    

if __name__ == "__main__":
    import uvicorn
    print("Starting Text-to-Image Retrieval API...")
    print("API Documentation available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)