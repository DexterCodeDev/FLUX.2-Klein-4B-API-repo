import os
import torch
import gradio as gr
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

# Fetch the Hugging Face token passed securely from Cloud Run Secrets
hf_token = os.environ.get("HF_TOKEN")
model_id = "black-forest-labs/FLUX.2-klein-4B"

print("Loading Text-to-Image Pipeline...")
# Load the base Text-to-Image pipeline in bfloat16 to fit safely in the L4 GPU
pipe_t2i = AutoPipelineForText2Image.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    token=hf_token
).to("cuda")

print("Loading Image-to-Image Pipeline...")
# Initialize the Image-to-Image pipeline by sharing the weights from the T2I pipeline.
# This prevents memory conflicts and Out-Of-Memory (OOM) errors.
pipe_i2i = AutoPipelineForImage2Image.from_pipe(pipe_t2i)

def generate_image(prompt, reference_image, guidance_scale, steps):
    if reference_image is None:
        # Execute Text-to-Image
        result = pipe_t2i(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps
        ).images[0]
    else:
        # Execute Image-to-Image
        result = pipe_i2i(
            prompt=prompt,
            image=reference_image,
            guidance_scale=guidance_scale,
            num_inference_steps=steps
        ).images[0]
    return result

# Build the UI
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# FLUX.2 [Klein] 4B Studio")
    gr.Markdown("Supports **Text-to-Image** and **Image-to-Image**. Upload an image below to switch to Image-to-Image mode.")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Describe what you want to generate...")
            reference_image = gr.Image(label="Reference Image (Optional)", type="pil")
            
            with gr.Accordion("Advanced Settings", open=False):
                guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=10.0, value=2.5, step=0.1)
                steps = gr.Slider(label="Inference Steps", minimum=1, maximum=50, value=20, step=1)
                
            generate_btn = gr.Button("Generate", variant="primary")
            
        with gr.Column():
            output_image = gr.Image(label="Output Image")
            
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, reference_image, guidance_scale, steps],
        outputs=output_image
    )

if __name__ == "__main__":
    # Cloud Run dynamically assigns a port via the PORT environment variable (defaulting to 8080)
    port = int(os.environ.get("PORT", 8080))
    app.launch(server_name="0.0.0.0", server_port=port)
