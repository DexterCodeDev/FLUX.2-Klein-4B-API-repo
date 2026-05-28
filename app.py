```python
import os
import io
import base64
from typing import Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from diffusers import DiffusionPipeline

# -----------------------------
# Config
# -----------------------------

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
HF_TOKEN = os.getenv("HF_TOKEN")

app = FastAPI(title="FLUX.2 Klein API")

pipe = None


# -----------------------------
# Model Loader
# -----------------------------

def load_model():
    global pipe

    if pipe is not None:
        return pipe

    print("Loading FLUX.2-klein-4B model...")

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
    )

    pipe.to("cuda")

    # Reduce VRAM spikes
    pipe.enable_attention_slicing()

    print("Model loaded successfully.")

    return pipe


# -----------------------------
# Request Models
# -----------------------------

class Txt2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    width: Optional[int] = 768
    height: Optional[int] = 1024
    num_inference_steps: Optional[int] = 28
    guidance_scale: Optional[float] = 3.5
    seed: Optional[int] = None


class Img2ImgRequest(BaseModel):
    prompt: str
    image_base64: str
    strength: Optional[float] = 0.75
    width: Optional[int] = 768
    height: Optional[int] = 1024
    num_inference_steps: Optional[int] = 28
    guidance_scale: Optional[float] = 3.5
    seed: Optional[int] = None


# -----------------------------
# Utilities
# -----------------------------

def pil_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# -----------------------------
# Routes
# -----------------------------

@app.get("/")
async def root():
    return {"status": "running"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/txt2img")
async def txt2img(request: Txt2ImgRequest):
    pipe = load_model()

    generator = None
    if request.seed is not None:
        generator = torch.Generator("cuda").manual_seed(
            request.seed
        )

    result = pipe(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        guidance_scale=request.guidance_scale,
        num_inference_steps=request.num_inference_steps,
        generator=generator,
    )

    images = [
        pil_to_base64(img)
        for img in result.images
    ]

    return {
        "images": images
    }


@app.post("/img2img")
async def img2img(request: Img2ImgRequest):
    pipe = load_model()

    image = Image.open(
        io.BytesIO(
            base64.b64decode(
                request.image_base64
            )
        )
    ).convert("RGB")

    generator = None
    if request.seed is not None:
        generator = torch.Generator("cuda").manual_seed(
            request.seed
        )

    result = pipe(
        prompt=request.prompt,
        image=image,
        strength=request.strength,
        width=request.width,
        height=request.height,
        guidance_scale=request.guidance_scale,
        num_inference_steps=request.num_inference_steps,
        generator=generator,
    )

    images = [
        pil_to_base64(img)
        for img in result.images
    ]

    return {
        "images": images
    }
```
