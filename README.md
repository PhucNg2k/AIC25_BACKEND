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




### Running the API
```bash
cd AIC25_BACKEND
python API/main_api.py
```

### API Documentation
- Interactive docs: `http://localhost:8000/docs`

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
