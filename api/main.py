
from fastapi import FastAPI, UploadFile
import logging
from PIL import Image
import io

from models.schemas import QueryRequest, DocumentResponse, SelectRequest, CaptionRequest
from services.search import get_docs
from utils.data import log_event
app = FastAPI()

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the search API. Use POST /search to send queries."}

@app.post("/search", response_model=DocumentResponse)
async def search(query_request: QueryRequest):
    query = query_request.query
    rel_docs, urls, distances, indices  = get_docs(query)

    # Convert indices and distances to lists if they are not already
    indices = indices.tolist() if hasattr(indices, 'tolist') else [int(i) for i in indices]
    distances = distances.tolist() if hasattr(distances, 'tolist') else [float(d) for d in distances]

    log_event("search", query, indices)

    return {
        "rel_docs": rel_docs,
        "urls": urls,
        "rel_docs_sim": distances,
        "indices": indices
    }

@app.post("/process-image")
async def process_image(image: UploadFile):
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents))
    logger.debug(f"Image mode: {pil_image.mode}")
    logger.debug(f"Image size: {pil_image.size}")
    logger.debug(f"Image format: {pil_image.format}") 
    # TODO: pass PIL image to model
    return {"caption": "my caption"}

@app.post("/select")
async def select(select_request: SelectRequest):
    event_type = select_request.event_type
    query = select_request.query
    selected_doc_id = select_request.selected_doc_id

    print(query, selected_doc_id)
    log_event(event_type, query, selected_doc_id)
    return {"message": f"{event_type} event logged"} # ??

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
