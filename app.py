from io import BytesIO
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from inference import generate_image, warmup_model

app = FastAPI(
    title="FLUX.2 Klein 4B API",
    version="1.0.0"
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)

    negative_prompt: Optional[str] = ""

    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)

    steps: int = Field(default=4, ge=1, le=50)
    guidance_scale: float = Field(default=3.5, ge=0.0, le=20.0)

    seed: Optional[int] = None

    output_format: str = Field(default="webp")


@app.on_event("startup")
def startup_event():
    """
    Load model once at startup.
    Prevents reloading on every request.
    """
    warmup_model()


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "flux2-klein-4b"
    }


@app.post("/generate")
def generate(payload: GenerateRequest):

    try:
        image = generate_image(
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt,
            width=payload.width,
            height=payload.height,
            steps=payload.steps,
            guidance_scale=payload.guidance_scale,
            seed=payload.seed,
        )

        image_bytes = BytesIO()

        fmt = payload.output_format.lower()

        if fmt not in ["png", "jpeg", "jpg", "webp"]:
            fmt = "webp"

        save_format = "JPEG" if fmt == "jpg" else fmt.upper()

        image.save(
            image_bytes,
            format=save_format,
            quality=95
        )

        media_type = {
            "png": "image/png",
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "webp": "image/webp"
        }[fmt]

        return Response(
            content=image_bytes.getvalue(),
            media_type=media_type
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
