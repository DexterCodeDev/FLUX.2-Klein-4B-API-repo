import io
import os
import random
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
from fastapi import (
    FastAPI,
    File,
    Form,
    UploadFile,
)
from fastapi.responses import JSONResponse, StreamingResponse
from huggingface_hub import login
from PIL import Image

from diffusers import DiffusionPipeline


MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"

pipe = None


QUALITY_PRESETS = {
    "fast": {
        "steps": 4,
        "guidance_scale": 1.0,
    },
    "balanced": {
        "steps": 12,
        "guidance_scale": 2.5,
    },
    "high": {
        "steps": 28,
        "guidance_scale": 3.5,
    },
    "cost-aware": {
        "steps": 8,
        "guidance_scale": 1.8,
    }
}


def pil_to_bytes(image: Image.Image, fmt="PNG"):
    output = io.BytesIO()
    image.save(output, format=fmt)
    output.seek(0)
    return output


def normalize_output_format(fmt: str):
    fmt = fmt.lower()

    if fmt in ["png", "jpg", "jpeg", "webp"]:
        return fmt

    return "png"


def load_image(file_bytes: bytes):
    return Image.open(
        io.BytesIO(file_bytes)
    ).convert("RGB")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe

    hf_token = os.getenv("HF_TOKEN")

    if hf_token:
        print("Logging into HuggingFace...")
        login(token=hf_token)

    print("Loading FLUX model...")

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available()
        else torch.float32
    )

    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )

    if torch.cuda.is_available():
        pipe.to("cuda")

        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

    print("FLUX pipeline loaded.")

    yield


app = FastAPI(
    title="FLUX API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    return {
        "message": "FLUX API running"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }


@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    width: int = Form(1024),
    height: int = Form(1024),
    quality: str = Form("balanced"),
    steps: Optional[int] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    seed: Optional[int] = Form(None),
    num_images: int = Form(1),
    output_format: str = Form("png"),
):
    try:
        global pipe

        preset = QUALITY_PRESETS.get(
            quality,
            QUALITY_PRESETS["balanced"]
        )

        if steps is None:
            steps = preset["steps"]

        if guidance_scale is None:
            guidance_scale = preset["guidance_scale"]

        if seed is None:
            seed = random.randint(
                0,
                999999999
            )

        generator = torch.Generator(
            device="cuda"
            if torch.cuda.is_available()
            else "cpu"
        ).manual_seed(seed)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
        )

        image = result.images[0]

        fmt = normalize_output_format(
            output_format
        )

        output = pil_to_bytes(
            image,
            fmt.upper()
        )

        media_type = (
            f"image/{fmt}"
        )

        return StreamingResponse(
            output,
            media_type=media_type,
            headers={
                "X-Seed": str(seed)
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )


@app.post("/edit")
async def edit(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    negative_prompt: str = Form(""),
    width: int = Form(1024),
    height: int = Form(1024),
    quality: str = Form("balanced"),
    steps: Optional[int] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    strength: float = Form(0.8),  # experimental
    seed: Optional[int] = Form(None),
    output_format: str = Form("png"),
):
    try:
        global pipe

        preset = QUALITY_PRESETS.get(
            quality,
            QUALITY_PRESETS["balanced"]
        )

        if steps is None:
            steps = preset["steps"]

        if guidance_scale is None:
            guidance_scale = preset["guidance_scale"]

        if seed is None:
            seed = random.randint(
                0,
                999999999
            )

        image_bytes = await image.read()

        input_image = load_image(
            image_bytes
        )

        generator = torch.Generator(
            device="cuda"
            if torch.cuda.is_available()
            else "cpu"
        ).manual_seed(seed)

        kwargs = dict(
            prompt=prompt,
            image=input_image,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        # Experimental support
        try:
            kwargs["strength"] = strength
        except Exception:
            pass

        result = pipe(**kwargs)

        image = result.images[0]

        fmt = normalize_output_format(
            output_format
        )

        output = pil_to_bytes(
            image,
            fmt.upper()
        )

        media_type = (
            f"image/{fmt}"
        )

        return StreamingResponse(
            output,
            media_type=media_type,
            headers={
                "X-Seed": str(seed)
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )


@app.post("/advanced")
async def advanced(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    image: UploadFile = File(None),
    reference_images: List[UploadFile] = File([]),
    scheduler: str = Form("default"),
    width: int = Form(1024),
    height: int = Form(1024),
    quality: str = Form("balanced"),
    steps: int = Form(12),
    guidance_scale: float = Form(2.5),
    strength: float = Form(0.8),
    seed: Optional[int] = Form(None),
    output_format: str = Form("png"),
):
    try:
        global pipe

        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
        )

        if image:
            image_bytes = await image.read()

            kwargs["image"] = load_image(
                image_bytes
            )

            kwargs["strength"] = strength

        # EXPERIMENTAL:
        # guessed multi-reference behavior
        if reference_images:
            refs = []

            for ref in reference_images:
                ref_bytes = await ref.read()

                refs.append(
                    load_image(ref_bytes)
                )

            kwargs["reference_images"] = refs

        if seed is None:
            seed = random.randint(
                0,
                999999999
            )

        generator = torch.Generator(
            device="cuda"
            if torch.cuda.is_available()
            else "cpu"
        ).manual_seed(seed)

        kwargs["generator"] = generator

        result = pipe(**kwargs)

        image = result.images[0]

        fmt = normalize_output_format(
            output_format
        )

        output = pil_to_bytes(
            image,
            fmt.upper()
        )

        return StreamingResponse(
            output,
            media_type=f"image/{fmt}",
            headers={
                "X-Seed": str(seed)
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e)
            }
        )
