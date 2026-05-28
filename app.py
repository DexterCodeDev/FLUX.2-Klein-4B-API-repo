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
model_error = None


class GenerateRequest(BaseModel):
    prompt: str
    image_base64: str | None = None
    strength: float = 0.75
    guidance_scale: float = 1.0
    num_inference_steps: int = 4
    width: int = 1024
    height: int = 1024
    seed: int | None = None


def load_model():

    global txt2img_pipe
    global img2img_pipe
    global model_loading
    global model_loaded
    global model_error

    try:

        print("STARTING MODEL LOAD")

        model_loading = True

        hf_token = os.getenv("HF_TOKEN")

        txt2img_pipe = FluxPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            token=hf_token
        )

        print("TXT2IMG PIPELINE LOADED")

        img2img_pipe = FluxImg2ImgPipeline.from_pipe(
            txt2img_pipe
        )

        print("IMG2IMG PIPELINE CREATED")

        txt2img_pipe.enable_model_cpu_offload()

        txt2img_pipe.vae.enable_slicing()
        txt2img_pipe.vae.enable_tiling()

        txt2img_pipe.set_progress_bar_config(
            disable=True
        )

        img2img_pipe.set_progress_bar_config(
            disable=True
        )

        model_loaded = True
        model_loading = False

        print("MODEL LOADED SUCCESSFULLY")

    except Exception as e:

        model_loading = False
        model_loaded = False

        model_error = str(e)

        print("MODEL LOAD FAILED")
        print(str(e))


@app.on_event("startup")
async def startup_event():

    threading.Thread(
        target=load_model,
        daemon=True
    ).start()


@app.get("/")
async def root():

    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_loading": model_loading,
        "model_error": model_error
    }


@app.post("/generate")
async def generate(req: GenerateRequest):

    global txt2img_pipe
    global img2img_pipe

    if model_error:

        return {
            "status": "error",
            "message": model_error
        }

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
