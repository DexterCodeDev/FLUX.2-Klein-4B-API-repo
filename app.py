import os
import io
import time
import threading
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from inference import generate_image, warmup_model

app = FastAPI(title="FLUX.2 Klein 4B API")

# -----------------------------
# Global model loading state
# -----------------------------
model_ready = False
model_loading = False


# -----------------------------
# Request schema
# -----------------------------
class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    seed: int | None = None
    response_format: str = "png"  # png or base64


# -----------------------------
# Background model loading
# -----------------------------
def load_model():
    global model_ready, model_loading

    try:
        print("Loading FLUX.2 Klein 4B model...")
        warmup_model()
        model_ready = True
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Model loading failed: {e}")
    finally:
        model_loading = False


# -----------------------------
# Startup event
# -----------------------------
@app.on_event("startup")
async def startup_event():
    global model_loading

    if not model_loading:
        model_loading = True

        # Start model loading in background
        thread = threading.Thread(target=load_model)
        thread.daemon = True
        thread.start()


# -----------------------------
# Health route
# -----------------------------
@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_ready": model_ready,
        "model_loading": model_loading
    }


# -----------------------------
# Generate image endpoint
# -----------------------------
@app.post("/generate")
async def generate(req: GenerateRequest):
    global model_ready

    start = time.time()

    # Wait for model if still loading
    wait_seconds = 0
    while not model_ready:
        time.sleep(2)
        wait_seconds += 2

        if wait_seconds > 1800:
            raise HTTPException(
                status_code=503,
                detail="Model is still loading. Try again later."
            )

    try:
        image = generate_image(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            seed=req.seed
        )

        inference_time = round(time.time() - start, 2)

        img_io = io.BytesIO()
        image.save(img_io, format="PNG")
        img_io.seek(0)

        headers = {
            "X-Inference-Time": str(inference_time)
        }

        return StreamingResponse(
            img_io,
            media_type="image/png",
            headers=headers
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Run locally / Cloud Run
# -----------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        workers=1
    )
