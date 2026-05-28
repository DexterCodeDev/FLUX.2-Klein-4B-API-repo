import os
import io
import base64
import threading

import torch

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

from diffusers import (
    FluxPipeline,
    FluxImg2ImgPipeline
)

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"

app = FastAPI()

txt2img_pipe = None
img2img_pipe = None

model_loading = False
model_loaded = False


class GenerateRequest(BaseModel):
    prompt: str
    image_base64: str | None = None
    strength: float = 0.75
    guidance_scale: float = 1.0
    num_inference_steps: int = 4
    width: int = 1024
    height: int = 1024
    seed: int | None = None


def load_models():
    global txt2img_pipe
    global img2img_pipe
    global model_loading
    global model_loaded

    if model_loading or model_loaded:
        return

    model_loading = True

    hf_token = os.getenv("HF_TOKEN")

    txt2img_pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        token=hf_token
    )

    img2img_pipe = FluxImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        token=hf_token
    )

    txt2img_pipe.enable_model_cpu_offload()
    img2img_pipe.enable_model_cpu_offload()

    txt2img_pipe.vae.enable_slicing()
    txt2img_pipe.vae.enable_tiling()

    img2img_pipe.vae.enable_slicing()
    img2img_pipe.vae.enable_tiling()

    txt2img_pipe.set_progress_bar_config(disable=True)
    img2img_pipe.set_progress_bar_config(disable=True)

    model_loaded = True
    model_loading = False

    print("MODEL LOADED")


@app.on_event("startup")
async def startup_event():
    threading.Thread(
        target=load_models,
        daemon=True
    ).start()


@app.get("/")
async def root():
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_loading": model_loading
    }


@app.post("/generate")
async def generate(req: GenerateRequest):

    global txt2img_pipe
    global img2img_pipe

    if not model_loaded:
        return {
            "status": "loading_model"
        }

    try:

        generator = None

        if req.seed is not None:
            generator = torch.Generator(
                "cuda"
            ).manual_seed(req.seed)

        input_image = None

        if req.image_base64:

            image_data = base64.b64decode(
                req.image_base64
            )

            input_image = Image.open(
                io.BytesIO(image_data)
            ).convert("RGB")

            input_image = input_image.resize(
                (req.width, req.height)
            )

        if input_image is not None:

            result = img2img_pipe(
                prompt=req.prompt,
                image=input_image,
                strength=req.strength,
                guidance_scale=req.guidance_scale,
                num_inference_steps=req.num_inference_steps,
                generator=generator
            )

        else:

            result = txt2img_pipe(
                prompt=req.prompt,
                guidance_scale=req.guidance_scale,
                num_inference_steps=req.num_inference_steps,
                width=req.width,
                height=req.height,
                generator=generator
            )

        output_image = result.images[0]

        buffer = io.BytesIO()

        output_image.save(
            buffer,
            format="PNG"
        )

        output_base64 = base64.b64encode(
            buffer.getvalue()
        ).decode()

        return {
            "status": "success",
            "image_base64": output_base64
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
