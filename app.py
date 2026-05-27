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
def generate(
    prompt: str = Form(...),
    person_image: UploadFile = File(...),
    garment_image: UploadFile = File(...)
):
    # 1. Read files synchronously using file.file.read() to avoid async blocks
    person_bytes = person_image.file.read()
    garment_bytes = garment_image.file.read()

    # 2. Convert raw bytes to PIL Images
    person_pil = Image.open(BytesIO(person_bytes)).convert("RGB")
    garment_pil = Image.open(BytesIO(garment_bytes)).convert("RGB")

    # 3. Pass images and prompt directly to your model
    output_image = generate_image(
        prompt=prompt, 
        person_img=person_pil, 
        get_model=garment_pil
    )

    # 4. Stream the output image back cleanly
    img_io = BytesIO()
    output_image.save(img_io, format="PNG")
    img_io.seek(0)

    return StreamingResponse(
        img_io,
        media_type="image/png"
    )
