# AIC25 Backend API Documentation

## Overview

The AIC25 Backend API is a FastAPI-based service that provides multimodal search capabilities for video keyframes using CLIP embeddings, FAISS indexing, and Elasticsearch. The API supports text-to-image retrieval, OCR search, ASR search, and CSV submission functionality for competition submissions.

## Architecture

### Core Components

- **FastAPI Application**: Main API server with CORS middleware
- **CLIP Model**: Vision-Language model for text-to-image similarity search
- **FAISS Index**: Vector similarity search engine for fast retrieval
- **Elasticsearch**: Full-text search for OCR and ASR data
- **Multi-stage Search**: Supports complex multi-modal search workflows

### Key Technologies

- **FastAPI** 0.116.1 - Web framework
- **PyTorch** 2.8.0 - Deep learning framework
- **FAISS** 1.12.0 - Vector similarity search
- **Elasticsearch** 9.1.0 - Full-text search engine
- **OpenCLIP** 3.1.0 - CLIP model implementation

## API Endpoints

### Base URL
```
http://localhost:8000
```

### Health & Status

#### `GET /`
- **Description**: Root endpoint
- **Response**: Basic API status
- **Example Response**:
```json
{
  "message": "Text-to-Image Retrieval API is running",
  "status": "healthy"
}
```

#### `GET /health`
- **Description**: Detailed health check
- **Response**: System status including index and metadata status
- **Example Response**:
```json
{
  "status": "healthy",
  "index_status": "loaded",
  "metadata_status": "loaded",
  "total_images": 75011
}
```

### Search Endpoints

#### `POST /search/text`
- **Description**: Text-based image search using CLIP embeddings
- **Request Body**:
```json
{
  "value": "search query text",
  "top_k": 30
}
```
- **Response**: `SearchResponse` with matching keyframes

#### `POST /search/image`
- **Description**: Image-based similarity search
- **Request**: Multipart form with image file
- **Parameters**:
  - `image_file`: Image file (required)
  - `top_k`: Number of results (default: 30)
- **Response**: `SearchResponse` with similar keyframes

### Elasticsearch Search Endpoints

#### `POST /es-search/ocr`
- **Description**: OCR text search in video frames
- **Request Body**:
```json
{
  "value": "text to search",
  "top_k": 50
}
```
- **Response**: `SearchResponse` with OCR matches

#### `POST /es-search/video_name`
- **Description**: Search by video name with fuzzy matching
- **Request Body**:
```json
{
  "value": "video name",
  "top_k": 100
}
```
- **Response**: `SearchResponse` with video name matches

### Multi-Stage Search

#### `POST /search-entry`
- **Description**: Advanced multi-modal search with multiple stages
- **Request**: Multipart form data
- **Parameters**:
  - `stage_list`: JSON string defining search stages
  - `top_k`: Number of results per stage
- **Features**:
  - Multi-stage search processing
  - Event chaining for temporal sequences
  - Weighted result fusion
  - Duplicate frame removal

### CSV Submission Endpoints

#### `POST /submitCSV/kis`
- **Description**: Create CSV submission for KIS task
- **Request Body**:
```json
{
  "query_id": "unique_query_identifier",
  "query_str": "original query string",
  "selected_frames": ["video1_frame1", "video2_frame2"]
}
```
- **Response**: `MakeCSV_Response` with submission details

#### `POST /submitCSV/qa`
- **Description**: Create CSV submission for QA task
- **Request Body**:
```json
{
  "query_id": "unique_query_identifier",
  "query_str": "original query string",
  "qa_data": [
    {
      "video_name": "video1",
      "frame_idx": 123,
      "answer": "answer text"
    }
  ]
}
```
- **Response**: `MakeCSV_Response` with submission details

#### `POST /submitCSV/trake`
- **Description**: Create CSV submission for TRAKE task
- **Request Body**:
```json
{
  "query_id": "unique_query_identifier",
  "query_str": "original query string",
  "trake_data": [
    {
      "video_name": "video1",
      "frames": [123, 124, 125]
    }
  ]
}
```
- **Response**: `MakeCSV_Response` with submission details

## Data Models

### Core Response Models

#### `ImageResult`
```json
{
  "video_name": "string",
  "frame_idx": 123,
  "image_path": "string",
  "pts_time": 12.34,
  "score": 0.85
}
```

#### `SearchResponse`
```json
{
  "success": true,
  "query": "search query",
  "results": [ImageResult],
  "total_results": 25,
  "message": "Found 25 results"
}
```

### Request Models

#### `SearchRequest`
```json
{
  "value": "search text",
  "top_k": 30
}
```

#### `SearchEntryRequest`
```json
{
  "stage_list": {
    "stage1": {
      "text": {"value": "text query"},
      "ocr": {"value": "ocr query"},
      "img": {"value": "image_field_name"},
      "weight_dict": {"text": 0.5, "ocr": 0.3, "img": 0.2}
    }
  },
  "top_k": 100
}
```

### CSV Submission Models

#### `KIS_rq_CSV`
```json
{
  "query_id": "string",
  "query_str": "string",
  "selected_frames": ["string"]
}
```

#### `QA_CSV_Request`
```json
{
  "query_id": "string",
  "query_str": "string",
  "qa_data": [
    {
      "video_name": "string",
      "frame_idx": 123,
      "answer": "string"
    }
  ]
}
```

#### `TRAKE_CSV_Request`
```json
{
  "query_id": "string",
  "query_str": "string",
  "trake_data": [
    {
      "video_name": "string",
      "frames": [123, 124, 125]
    }
  ]
}
```

## Configuration

### Environment Variables

- `ES_LOCAL_URL`: Elasticsearch server URL
- `ES_LOCAL_API_KEY`: Elasticsearch API key

### Configuration Files

#### `config.py`
```python
DATA_SOURCE = '/REAL_DATA/keyframes_b1/keyframes'
FAISS_SAVE_DIR = "FaissIndex"
INDEX_SAVE_PATH = "faiss_index_vitL.bin"
METADATA_AVE_PATH = "id_to_name_vitL.json"
CLIP_EMBED_DIM = 1024
OCR_INDEX_NAME = 'ocr_index'
```

## Dependencies

### Core Dependencies
- `fastapi==0.116.1` - Web framework
- `torch==2.8.0` - Deep learning
- `faiss-cpu==1.12.0` - Vector search
- `elasticsearch==9.1.0` - Search engine
- `open_clip_torch==3.1.0` - CLIP model
- `pillow==11.3.0` - Image processing
- `pydantic==2.11.7` - Data validation

### Additional Dependencies
- `uvicorn==0.35.0` - ASGI server
- `python-multipart==0.0.20` - File uploads
- `httpx==0.28.1` - HTTP client
- `pandas==2.3.2` - Data processing

## Search Features

### Multi-Modal Search
- **Text Search**: CLIP-based semantic search
- **Image Search**: Visual similarity using CLIP embeddings
- **OCR Search**: Full-text search in extracted text
- **ASR Search**: Audio transcript search
- **Video Name Search**: Fuzzy video name matching

### Advanced Features
- **Temporal Chaining**: Links frames across time sequences
- **Event Chaining**: Connects related events across stages
- **Weighted Fusion**: Combines multiple search modalities
- **Score Normalization**: Normalizes scores to [0,1] range
- **Duplicate Removal**: Removes duplicate frames by video+frame

### Search Processing Pipeline
1. **Stage Processing**: Each stage processes multiple modalities
2. **Async Execution**: Parallel processing of different modalities
3. **Weight Application**: Applies user-defined weights to results
4. **Temporal Linking**: Connects frames within time windows
5. **Event Chaining**: Links events across different stages
6. **Score Updates**: Updates scores based on temporal relationships
7. **Deduplication**: Removes duplicate frames
8. **Normalization**: Normalizes final scores

## Error Handling

### HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid input)
- `409`: Conflict (duplicate submission)
- `500`: Internal Server Error

### Error Response Format
```json
{
  "detail": "Error message description"
}
```

## Usage Examples

### Basic Text Search
```python
import requests

response = requests.post(
    "http://localhost:8000/search/text",
    json={"value": "person walking", "top_k": 10}
)
results = response.json()
```

### Image Search
```python
import requests

with open("query_image.jpg", "rb") as f:
    files = {"image_file": f}
    data = {"top_k": "10"}
    response = requests.post(
        "http://localhost:8000/search/image",
        files=files,
        data=data
    )
results = response.json()
```

### Multi-Stage Search
```python
import requests
import json

stage_data = {
    "stage1": {
        "text": {"value": "person"},
        "ocr": {"value": "sign"},
        "weight_dict": {"text": 0.7, "ocr": 0.3}
    }
}

data = {
    "stage_list": json.dumps(stage_data),
    "top_k": "50"
}

response = requests.post(
    "http://localhost:8000/search-entry",
    data=data
)
results = response.json()
```

## Development

### Running the API
```bash
cd AIC25_BACKEND
python API/main_api.py
```

### API Documentation
- Interactive docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Dependencies Installation
```bash
pip install -r requirements.txt
```

## File Structure

```
AIC25_BACKEND/
├── API/
│   ├── main_api.py              # Main FastAPI application
│   ├── routes/                  # API route modules
│   │   ├── search_routes.py     # Text/image search endpoints
│   │   ├── es_routes.py         # Elasticsearch endpoints
│   │   └── submit_csv_routes.py # CSV submission endpoints
│   ├── models/                  # Pydantic data models
│   │   ├── request.py           # Request models
│   │   ├── response.py          # Response models
│   │   ├── entry_models.py      # Multi-stage search models
│   │   └── csv.py               # CSV submission models
│   ├── ElasticSearch/           # Elasticsearch client
│   │   └── ESclient.py          # ES client implementation
│   ├── dependecies.py          # FastAPI dependencies
│   ├── utils.py                 # Utility functions
│   ├── results_utils.py         # Search result processing
│   └── search_utils.py          # Search orchestration
├── config.py                    # Configuration settings
├── retrieve_vitL.py            # CLIP/FAISS search implementation
├── model_loading.py            # Model loading utilities
└── requirements.txt            # Python dependencies
```

## Performance Considerations

- **FAISS Index**: Pre-built index for fast vector similarity search
- **Async Processing**: Parallel execution of multiple search modalities
- **Caching**: Model and index loaded once at startup
- **Batch Processing**: Efficient bulk operations for Elasticsearch
- **Memory Management**: Optimized tensor operations with GPU support

## Security

- **CORS**: Configured for cross-origin requests
- **Input Validation**: Pydantic models validate all inputs
- **File Upload Limits**: Controlled file size and type validation
- **Error Handling**: Secure error messages without sensitive information
