import torch
from diffusers import DiffusionPipeline
from config import settings

# Global model variable
pipe = None


def get_model():
    global pipe

    # Load only once (lazy loading)
    if pipe is None:
        print("Loading FLUX.2 Klein 4B model...")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        pipe = DiffusionPipeline.from_pretrained(
            settings.MODEL_ID,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )

        pipe.to(device)

        print("Model loaded successfully")

    return pipe


def generate_image(prompt: str):
    """
    Generate image from prompt
    """
    pipe = get_model()

    image = pipe(
        prompt=prompt
    ).images[0]

    return image


def warmup_model():
    """
    No-op to prevent startup loading.
    Keeps compatibility if app.py still imports it.
    """
    pass
