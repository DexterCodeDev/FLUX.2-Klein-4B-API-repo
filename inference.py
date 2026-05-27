# inference.py
import torch
from diffusers import StableDiffusionPipeline
import base64
import io
from PIL import Image
import random

# Load your model globally
model = None

def warmup_model():
    global model
    if model is None:
        model = StableDiffusionPipeline.from_pretrained(
            "path_to_your_model_or_hf_id",
            torch_dtype=torch.float16
        ).to("cuda")
        # Optional: generate a tiny image to initialize everything
        _ = model("warmup", num_inference_steps=1)

def generate_image(prompt, negative_prompt=None, width=512, height=512, steps=20,
                   guidance_scale=7.5, seed=None, num_images=1):
    global model
    if model is None:
        warmup_model()

    seed_used = seed or random.randint(0, 2**32-1)
    generator = torch.Generator("cuda").manual_seed(seed_used)

    outputs = []
    for _ in range(num_images):
        image = model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
        
        # Convert to Base64 for API response
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        outputs.append(img_str)

    return outputs, seed_used
