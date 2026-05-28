import io
from threading import Thread

import torch
from diffusers import FluxImg2ImgPipeline
from PIL import Image

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"

pipe = None
model_loading = False
model_ready = False


def load_model():
    global pipe, model_loading, model_ready

    if pipe is not None or model_loading:
        return

    model_loading = True

    try:
        print("Loading FLUX.2 Klein 4B model...")

        pipe = FluxImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16
        )

        pipe.enable_model_cpu_offload()

        model_ready = True
        print("Model loaded successfully!")

    except Exception as e:
        print(f"Model load failed: {e}")
        model_ready = False

    model_loading = False


def warmup_model():
    Thread(target=load_model, daemon=True).start()


def generate_image(
    prompt,
    image_bytes,
    negative_prompt=None,
    width=1024,
    height=1024,
    strength=0.75,
    num_inference_steps=30,
    guidance_scale=4.0,
    seed=None
):
    global pipe

    if pipe is None:
        raise Exception("Model not loaded yet")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    generator = None
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        image=image,
        strength=strength,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )

    output_image = result.images[0]

    img_bytes = io.BytesIO()
    output_image.save(img_bytes, format="PNG")

    return img_bytes.getvalue()
