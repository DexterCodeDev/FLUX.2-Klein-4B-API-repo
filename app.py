import io
import os
import base64
import torch

from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from diffusers import DiffusionPipeline
from huggingface_hub import login

# =========================================================
# CONFIG
# =========================================================

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable not found")

login(token=HF_TOKEN)

# =========================================================
# LOAD MODEL
# =========================================================

print("Loading model...")

pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()

print("Model loaded successfully")

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(title="FLUX.2-klein-4B API")


class GenerateRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int = 4
    guidance_scale: float = 1.0
    seed: int | None = None


class EditRequest(BaseModel):
    prompt: str
    image_base64: str
    steps: int = 4
    guidance_scale: float = 1.0
    seed: int | None = None


def pil_to_base64(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.get("/")
def root():
    return {
        "status": "running",
        "model": MODEL_ID
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/generate")
def generate(req: GenerateRequest):

    try:
        generator = None

        if req.seed is not None:
            generator = torch.Generator("cpu").manual_seed(req.seed)

        image = pipe(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            generator=generator,
        ).images[0]

        return {
            "image": pil_to_base64(image)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/edit")
def edit(req: EditRequest):

    try:
        image_data = base64.b64decode(req.image_base64)
        input_image = Image.open(io.BytesIO(image_data)).convert("RGB")

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    try:
        generator = None

        if req.seed is not None:
            generator = torch.Generator("cpu").manual_seed(req.seed)

        image = pipe(
            prompt=req.prompt,
            image=input_image,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance_scale,
            generator=generator,
        ).images[0]

        return {
            "image": pil_to_base64(image)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
