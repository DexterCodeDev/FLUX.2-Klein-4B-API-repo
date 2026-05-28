import os
import io
import base64
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import torch
from diffusers import DiffusionPipeline

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"

app = FastAPI(title="FLUX.2-klein-4B CloudRun API")

# Load model once globally
print("Loading FLUX.2-klein-4B model...")
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    use_auth_token=HF_TOKEN,
    safety_checker=None
)
pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()
print("Model loaded successfully.")


# ----------- Request Models -----------

class Txt2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    width: Optional[int] = 768
    height: Optional[int] = 768
    num_inference_steps: Optional[int] = 28
    guidance_scale: Optional[float] = 7.5
    seed: Optional[int] = None
    num_images: Optional[int] = 1


class Img2ImgRequest(BaseModel):
    prompt: str
    image_base64: str
    strength: Optional[float] = 0.75
    width: Optional[int] = 768
    height: Optional[int] = 768
    num_inference_steps: Optional[int] = 28
    guidance_scale: Optional[float] = 7.5
    seed: Optional[int] = None
    num_images: Optional[int] = 1


# ----------- Utility Function -----------

def pil_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


# ----------- Endpoints -----------

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/txt2img")
async def txt2img(request: Txt2ImgRequest):
    generator = torch.Generator("cuda")
    if request.seed is not None:
        generator = generator.manual_seed(request.seed)
    images = pipe(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        generator=generator,
    ).images

    base64_images = [pil_to_base64(img) for img in images]
    return {"images": base64_images}


@app.post("/img2img")
async def img2img(request: Img2ImgRequest):
    input_image = Image.open(io.BytesIO(base64.b64decode(request.image_base64))).convert("RGB")

    generator = torch.Generator("cuda")
    if request.seed is not None:
        generator = generator.manual_seed(request.seed)

    images = pipe(
        prompt=request.prompt,
        image=input_image,
        strength=request.strength,
        width=request.width,
        height=request.height,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        generator=generator,
    ).images

    base64_images = [pil_to_base64(img) for img in images]
    return {"images": base64_images}
