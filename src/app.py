from typing import Dict, List, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from service import txtai_service
from config import Config

app = FastAPI(title="TxtAI Service")


class Document(BaseModel):
    id: str
    text: str


class BulkIndexRequest(BaseModel):
    documents: List[Document]


class SearchRequest(BaseModel):
    text: str
    limit: int = 10
    offset: int = 0


@app.get("/info")
async def info():
    """Get information about the search service"""
    return txtai_service.get_info()


@app.post("/index")
async def index(document: Document):
    """Index a single document"""
    if not txtai_service.available:
        raise HTTPException(status_code=503, detail="Search service is not available")
    
    result = txtai_service.index_document(document.id, document.text)
    if result:
        return {"status": "success"}
    else:
        raise HTTPException(status_code=500, detail="Failed to index document")


@app.post("/bulk-index")
async def bulk_index(request: BulkIndexRequest):
    """Index multiple documents at once"""
    if not txtai_service.available:
        raise HTTPException(status_code=503, detail="Search service is not available")
    
    result = txtai_service.bulk_index([doc.dict() for doc in request.documents])
    if result.get("status") == "success":
        return result
    else:
        raise HTTPException(status_code=500, detail=result.get("message", "Failed to index documents"))


@app.post("/search")
async def search(request: SearchRequest):
    """Search for documents"""
    if not txtai_service.available:
        raise HTTPException(status_code=503, detail="Search service is not available")
    
    results = txtai_service.search(request.text, request.limit, request.offset)
    return {"results": results}


if __name__ == "__main__":
    uvicorn.run("app:app", host=Config.HOST, port=Config.PORT, reload=False)