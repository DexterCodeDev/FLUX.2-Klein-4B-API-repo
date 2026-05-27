from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from io import BytesIO

from inference import generate_image

app = FastAPI()


class ImageRequest(BaseModel):
    prompt: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: ImageRequest):
    image = generate_image(req.prompt)

    img_io = BytesIO()
    image.save(img_io, format="PNG")
    img_io.seek(0)

    return StreamingResponse(
        img_io,
        media_type="image/png"
    )
