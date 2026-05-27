import torch
from diffusers import Flux2KleinPipeline

pipe = None

device = "cuda"
dtype = torch.bfloat16


def warmup_model():
    global pipe

    if pipe is not None:
        return

    print("Loading FLUX.2 Klein 4B model...")

    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-base-4B",
        torch_dtype=dtype
    )

    pipe.enable_model_cpu_offload()

    print("Model loaded successfully!")


def generate_image(
    prompt,
    width=1024,
    height=1024,
    num_inference_steps=28,
    guidance_scale=4.0,
    seed=None,
    negative_prompt=None
):
    global pipe

    if pipe is None:
        warmup_model()

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator
    )

    return result.images[0]
