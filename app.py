from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from inference import (
    generate_image,
    warmup_model,
    model_ready,
    model_loading,
)

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    warmup_model()


@app.get("/")
def root():
    return {
        "status": "ok",
        "model_ready": model_ready,
        "model_loading": model_loading,
    }


@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    negative_prompt: str = Form(None),
    width: int = Form(1024),
    height: int = Form(1024),
    strength: float = Form(0.75),
    num_inference_steps: int = Form(30),
    guidance_scale: float = Form(4.0),
    seed: int = Form(None),
):
    try:
        image_bytes = await image.read()

        result = generate_image(
            prompt=prompt,
            image_bytes=image_bytes,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

        return Response(result, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
