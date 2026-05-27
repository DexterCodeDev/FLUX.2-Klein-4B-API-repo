import gc
from typing import Optional

import torch
from diffusers import DiffusionPipeline

from config import settings

pipe = None


def warmup_model():
    global pipe

    if pipe is not None:
        return pipe

    print("Loading FLUX.2 Klein 4B model...")

    pipe = DiffusionPipeline.from_pretrained(
        settings.MODEL_ID,
        torch_dtype=torch.bfloat16,
    )

    pipe.to("cuda")

    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    print("Model loaded successfully")

    return pipe


@torch.inference_mode()
def generate_image(
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 4,
    guidance_scale: float = 3.5,
    seed: Optional[int] = None,
):

    global pipe

    if pipe is None:
        warmup_model()

    generator = None

    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        generator=generator,
    )

    image = result.images[0]

    torch.cuda.empty_cache()
    gc.collect()

    return image
