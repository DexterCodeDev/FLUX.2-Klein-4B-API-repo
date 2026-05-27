# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from inference import generate_image, warmup_model
import time

app = FastAPI(title="FLUX.2 Klein 4B API", version="1.0.0")

# Warmup the model on start
@app.on_event("startup")
async def startup_event():
    warmup_model()

# Request schema for full-feature API
class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    width: Optional[int] = 512
    height: Optional[int] = 512
    steps: Optional[int] = 20
    guidance_scale: Optional[float] = 7.5
    seed: Optional[int] = None
    num_images: Optional[int] = 1

# Response schema
class ImageGenerationResponse(BaseModel):
    status: str
    generation_time: float
    seed: int
    images: List[str]  # Base64 strings

@app.post("/generate", response_model=ImageGenerationResponse)
async def generate(request: ImageGenerationRequest):
    start_time = time.time()
    try:
        images, seed_used = generate_image(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            num_images=request.num_images
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    generation_time = time.time() - start_time
    return ImageGenerationResponse(
        status="success",
        generation_time=generation_time,
        seed=seed_used,
        images=images
    )

# Health check
@app.get("/health")
async def health():
    return {"status": "ok"}
