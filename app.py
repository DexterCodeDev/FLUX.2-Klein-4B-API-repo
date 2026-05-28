import os
import torch
import gradio as gr
from diffusers import Flux2KleinPipeline

# Activate high-speed Rust downloads to speed up the container boot time
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

hf_token = os.environ.get("HF_TOKEN")
model_id = "black-forest-labs/FLUX.2-klein-4B"

print("Loading Unified FLUX.2 Klein Pipeline...")
# FLUX.2 Klein unifies generation and editing in a single pipeline
pipe = Flux2KleinPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    token=hf_token
).to("cuda")

def generate_image(prompt, reference_image, guidance_scale, steps):
    if reference_image is None:
        # Execute Text-to-Image
        result = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=torch.Generator(device="cuda").manual_seed(0)
        ).images[0]
    else:
        # Execute Image-to-Image by passing the image parameter
        result = pipe(
            prompt=prompt,
            image=reference_image,
            height=1024,
            width=1024,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=torch.Generator(device="cuda").manual_seed(0)
        ).images[0]
    return result

with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# FLUX.2 [Klein] 4B Studio")
    gr.Markdown("Supports **Text-to-Image** and **Image-to-Image**. Upload an image below to switch to Image-to-Image mode.")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Describe what you want to generate...")
            reference_image = gr.Image(label="Reference Image (Optional)", type="pil")
            
            with gr.Accordion("Advanced Settings", open=False):
                guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=10.0, value=1.0, step=0.1)
                steps = gr.Slider(label="Inference Steps", minimum=1, maximum=50, value=4, step=1)
                
            generate_btn = gr.Button("Generate", variant="primary")
            
        with gr.Column():
            output_image = gr.Image(label="Output Image")
            
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, reference_image, guidance_scale, steps],
        outputs=output_image
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.launch(server_name="0.0.0.0", server_port=port)
