import io
import os
import base64
import threading
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
# GLOBALS
# =========================================================

pipe = None
model_loading = True
model_error = None

# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(title="FLUX.2-klein-4B API")


# =========================================================
# REQUEST MODELS
# =========================================================

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


# =========================================================
# HELPERS
# =========================================================

def pil_to_base64(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# =========================================================
# MODEL LOADER
# =========================================================

def load_model():
    global pipe
    global model_loading
    global model_error

    try:
        print("Loading model...")

        pipe = DiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        pipe.enable_model_cpu_offload()
        pipe.enable_attention_slicing()

        print("Model loaded successfully")

        model_loading = False

    except Exception as e:
        model_error = str(e)
        model_loading = False
        print(f"MODEL LOAD ERROR: {e}")


# =========================================================
# START BACKGROUND MODEL LOADING
# =========================================================

threading.Thread(target=load_model, daemon=True).start()


# =========================================================
# ROUTES
# =========================================================

@app.get("/")
def root():
    return {
        "status": "running",
        "model_loaded": pipe is not None,
        "loading": model_loading,
        "error": model_error
    }


@app.get("/health")
def health():

    if model_error:
        return {
            "status": "error",
            "error": model_error
        }

    if model_loading:
        return {
            "status": "loading"
        }

    return {
        "status": "healthy"
    }


@app.post("/generate")
def generate(req: GenerateRequest):

    if model_loading:
        raise HTTPException(
            status_code=503,
            detail="Model still loading"
        )

    if model_error:
        raise HTTPException(
            status_code=500,
            detail=model_error
        )

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

    if model_loading:
        raise HTTPException(
            status_code=503,
            detail="Model still loading"
        )

    if model_error:
        raise HTTPException(
            status_code=500,
            detail=model_error
        )

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
