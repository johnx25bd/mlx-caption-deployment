
from fastapi import FastAPI, HTTPException, Request, UploadFile
import logging
from PIL import Image
import io

from services.caption import CaptionService
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
    return {"message": "Welcome to thecaption API. Use POST /process-image to send an image for captioning"}

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
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
