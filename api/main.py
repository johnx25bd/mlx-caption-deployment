
from datetime import datetime
import os
from pathlib import Path
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
import logging
from PIL import Image
import io
import uuid

from services.caption import CaptionService

UPLOAD_DIR = "app/images"
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
from data import save_file_to_disk, save_image_data_to_db

app = FastAPI()

logger = logging.getLogger('uvicorn.error')
logger.setLevel(logging.DEBUG)

caption_service = CaptionService()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}", exc_info=True)
        raise


@app.get("/")
async def read_root():
    return {"message": "Welcome to the caption API. Use POST /process-image to send an image for captioning"}

@app.post("/process-image")
async def process_image(image: UploadFile):
    logger.debug("Processing new image request")
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        if pil_image.mode != "RGB":
            logger.debug(f"Converting image from {pil_image.mode} to RGB")
            pil_image = pil_image.convert("RGB")
        caption = caption_service.predict_caption(pil_image)
        return {"caption": caption}
        
    except Exception as e:
        logger.error("Error in process_image", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/upload-image")
async def upload_image_handler(
    image: UploadFile = File(...),
    caption: str = Form(...)
):
    try:
        name, file_path = await save_file_to_disk(image)
        save_image_data_to_db(str(uuid.uuid4()), name, file_path, caption)
        return {"message": "Success"}
    except Exception as e:
        logger.error("Error in upload_image", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
