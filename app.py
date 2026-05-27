from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from io import BytesIO
import PIL.Image as Image

from inference import generate_image

app = FastAPI()


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...)
):
    # 1. Read files into memory as bytes
    person_bytes = await person_image.read()
    garment_bytes = await garment_image.read()

    # 2. Convert bytes to PIL Images
    person_pil = Image.open(BytesIO(person_bytes)).convert("RGB")
    garment_pil = Image.open(BytesIO(garment_bytes)).convert("RGB")

    # 3. Pass images and prompt to your model (Update inference.py to accept these!)
    output_image = generate_image(
        prompt=prompt, 
        person_img=person_pil, 
        garment_img=garment_pil
    )

    # 4. Stream the output image back to the client
    img_io = BytesIO()
    output_image.save(img_io, format="PNG")
    img_io.seek(0)

    return StreamingResponse(
        img_io,
        media_type="image/png"
    )
