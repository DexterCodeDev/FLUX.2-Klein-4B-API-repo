import torch
from diffusers import Flux2KleinInpaintPipeline  # Using the dedicated unified editing pipeline
from config import settings

# Global model variable
pipe = None


def get_model():
    global pipe

    # Load only once (lazy loading)
    if pipe is None:
        print("Loading FLUX.2 Klein 4B Inpaint/Edit model...")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Ensure your settings.MODEL_ID points to the edit/inpaint variant
        # e.g., "black-forest-labs/FLUX.2-klein-base-4B" or specialized try-on weights
        pipe = Flux2KleinInpaintPipeline.from_pretrained(
            settings.MODEL_ID,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        )

        # Enable memory optimizations for Cloud Run
        pipe.enable_model_cpu_offload() 
        print("Model loaded successfully")

    return pipe


def resize_for_flux(image, max_pixels=1024 * 1024):
    """
    Resizes image to stay within 1 Megapixel limits 
    and ensures dimensions are divisible by 16.
    """
    w, h = image.size
    total_pixels = w * h
    if total_pixels > max_pixels:
        scale = (max_pixels / total_pixels) ** 0.5
        w = int(w * scale)
        h = int(h * scale)
        
    # Snap to nearest multiple of 16 for VAE compatibility
    w = (w // 16) * 16
    h = (h // 16) * 16
    return image.resize((w, h))


def generate_image(prompt: str, person_img, garment_img):
    """
    Generate virtual try-on image using FLUX.2 Klein 4B multi-reference capabilities.
    """
    pipe = get_model()

    # Pre-process incoming PIL images from app.py
    person_resized = resize_for_flux(person_img)
    garment_resized = resize_for_flux(garment_img)

    # Execute pipeline matching the Multi-Reference input requirements
    # Note: Depending on your exact pipeline release, parameters may map to:
    # `image=person_resized`, `reference_image=garment_resized` 
    # OR passed together as a list of reference images.
    output = pipe(
        prompt=prompt,
        image=person_resized,
        mask_image=None,             # Can pass a skin/clothing mask if you generate one
        reference_image=garment_resized, 
        num_inference_steps=4,       # Optimized default step count for Klein distilled variants
        guidance_scale=3.5
    )

    return output.images[0]


def warmup_model():
    """
    Pre-loads model into memory during container startup
    """
    get_model()
