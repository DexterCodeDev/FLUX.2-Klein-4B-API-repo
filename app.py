import os
import io
import torch
import base64
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional
from PIL import Image
from diffusers import DiffusionPipeline

app = FastAPI(title="FLUX.2-klein-4B Unified Pipeline API")

# Global pipeline object
pipe = None

class GenerationRequest(BaseModel):
    prompt: str
    image_b64: Optional[str] = None  # Providing this triggers Image-to-Image mode
    strength: Optional[float] = 0.6  # Control over image changes (0.0 to 1.0)
    num_inference_steps: int = 4     # FLUX.2 Klein is 4-step distilled
    guidance_scale: float = 0.0      # Default for distilled variants

@app.on_event("startup")
def load_model():
    global pipe
    model_id = "black-forest-labs/FLUX.2-klein-4B"
    hf_token = os.getenv("HF_TOKEN")
    
    try:
        print("Loading Unified FLUX.2 [klein] 4B Pipeline onto L4 GPU...")
        # Using unified DiffusionPipeline to natively handle both modes dynamically
        pipe = DiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            token=hf_token
        )
        pipe.to("cuda")
        print("Pipeline initialized successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e

@app.post("/generate")
async def generate(request: GenerationRequest):
    global pipe
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")
    
    try:
        # Dictionary to store arguments dynamically
        pipeline_kwargs = {
            "prompt": request.prompt,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "generator": torch.Generator("cuda")
        }
        
        # Determine execution flow: Image-to-Image vs Text-to-Image
        if request.image_b64:
            print("Processing Mode: Image-to-Image")
            try:
                # Clear standard base64 headers if present (e.g. data:image/jpeg;base64,...)
                header_offset = request.image_b64.find(",") + 1 if "," in request.image_b64 else 0
                image_data = base64.b64decode(request.image_b64[header_offset:])
                input_image = Image.open(io.BytesIO(image_data)).convert("RGB")
                
                # Append image-specific inputs to unified pipeline execution
                pipeline_kwargs["image"] = input_image
                pipeline_kwargs["strength"] = request.strength
            except Exception as img_err:
                raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(img_err)}")
        else:
            print("Processing Mode: Text-to-Image")

        # Run inference using unified memory spaces 
        with torch.inference_mode():
            output = pipe(**pipeline_kwargs).images[0]
        
        # Save output image to byte buffer
        buffer = io.BytesIO()
        output.save(buffer, format="JPEG")
        buffer.seek(0)
        
        # Return as image stream
        return Response(content=buffer.getvalue(), media_type="image/jpeg")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "gpu_active": torch.cuda.is_available()}
