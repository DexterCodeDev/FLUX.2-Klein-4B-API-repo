import torch
from diffusers import DiffusionPipeline

pipe = None


def warmup_model():
    global pipe

    if pipe is not None:
        return

    print("Loading FLUX.2 Klein 4B model...")

    pipe = DiffusionPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-Klein-4B",
        torch_dtype=torch.bfloat16
    )

    pipe.enable_model_cpu_offload()

    print("Model loaded successfully!")


def generate_image(
    prompt,
    width=1024,
    height=1024,
    num_inference_steps=28,
    guidance_scale=3.5,
    seed=None,
    negative_prompt=None
):
    global pipe

    if pipe is None:
        warmup_model()

    generator = None
    if seed is not None:
        generator = torch.Generator("cpu").manual_seed(seed)

    kwargs = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
    }

    # FLUX models may not support negative prompt
    # so only send it if available
    if negative_prompt:
        try:
            result = pipe(
                **kwargs,
                negative_prompt=negative_prompt
            )
        except TypeError:
            result = pipe(**kwargs)
    else:
        result = pipe(**kwargs)

    image = result.images[0]
    return image
