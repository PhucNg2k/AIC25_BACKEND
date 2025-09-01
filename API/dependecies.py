from typing import Annotated
from fastapi import Depends, HTTPException
from retrieve_vitL import index, metadata

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

search_resource_Deps = Annotated[dict, Depends(get_search_resources)]