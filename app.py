import os
import torch
from diffusers import FluxPipeline, FluxImg2ImgPipeline
import gradio as gr
from huggingface_hub import login
from PIL import Image

# 1. Authentication & Model Setup
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("Warning: HF_TOKEN environment variable not found. Gated models might fail to load.")

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"

print(f"Loading model components from {MODEL_ID} into VRAM...")
# Load base Text-to-Image pipeline optimized for bfloat16 precision
txt2img_pipe = FluxPipeline.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16
)
txt2img_pipe.to("cuda")

# Create the Image-to-Image view by sharing the exact same underlying components
img2img_pipe = FluxImg2ImgPipeline.from_pipe(txt2img_pipe)
print("Model loaded successfully.")

# 2. Inference Logic
def generate_image(
    mode,
    prompt,
    init_image,
    strength,
    height,
    width,
    num_inference_steps,
    guidance_scale,
    seed
):
    # Set up generation seed for reproducibility
    generator = torch.Generator(device="cuda")
    if seed == -1:
        generator.seed()
    else:
        generator.manual_seed(int(seed))

    if mode == "Text to Image":
        output = txt2img_pipe(
            prompt=prompt,
            height=int(height),
            width=int(width),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=generator
        )
    else:  # Image to Image mode
        if init_image is None:
            raise gr.Error("Please upload an initial image for Image-to-Image generation.")
        
        # Pre-process image to correct dimensions if necessary
        input_img = Image.open(init_image).convert("RGB")
        input_img = input_img.resize((int(width), int(height)))

        output = img2img_pipe(
            prompt=prompt,
            image=input_img,
            strength=float(strength),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=generator
        )

    return output.images[0]

# 3. User Interface Assembly
# Using a custom soft theme for an organic, modern, flat UI aesthetic with smooth corner radii
with gr.Blocks(theme=gr.themes.Soft(radius_size="md", text_size="md")) as demo:
    gr.Markdown(
        """
        # FLUX.2 [Klein] 4B Studio
        Deploying high-fidelity image synthesis natively on Google Cloud Run.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            mode = gr.Radio(
                choices=["Text to Image", "Image to Image"], 
                value="Text to Image", 
                label="Generation Mode"
            )
            
            prompt = gr.Textbox(
                label="Prompt", 
                placeholder="Describe your vision completely here...", 
                lines=3
            )
            
            # Contextual input block for Image-to-Image conversions
            with gr.Group() as img2img_inputs:
                init_image = gr.Image(
                    label="Initial Image Source", 
                    type="filepath"
                )
                strength = gr.Slider(
                    minimum=0.0, 
                    maximum=1.0, 
                    value=0.75, 
                    step=0.05, 
                    label="Image Transformation Strength"
                )
            
            with gr.Accordion("Advanced Configuration Parameters", open=False):
                with gr.Row():
                    width = gr.Slider(minimum=256, maximum=1440, value=1024, step=64, label="Width")
                    height = gr.Slider(minimum=256, maximum=1440, value=1024, step=64, label="Height")
                
                num_inference_steps = gr.Slider(
                    minimum=1, maximum=100, value=28, step=1, label="Inference Steps"
                )
                guidance_scale = gr.Slider(
                    minimum=1.0, maximum=20.0, value=3.5, step=0.5, label="Guidance Scale"
                )
                seed = gr.Number(
                    value=-1, label="Randomization Seed (-1 for variations)"
                )
            
            generate_btn = gr.Button("Generate Masterpiece", variant="primary")
            
        with gr.Column(scale=1):
            output_image = gr.Image(label="Synthesized Output", interactive=False)

    # Simple toggle automation to manage interface visibility cleanly
    def update_ui_visibility(current_mode):
        if current_mode == "Text to Image":
            return gr.update(visible=False)
        return gr.update(visible=True)

    mode.change(
        fn=update_ui_visibility, 
        inputs=mode, 
        outputs=img2img_inputs
    )
    
    # Trigger visibility update right at load time
    demo.load(
        fn=lambda m: gr.update(visible=False) if m == "Text to Image" else gr.update(visible=True),
        inputs=mode,
        outputs=img2img_inputs
    )

    generate_btn.click(
        fn=generate_image,
        inputs=[
            mode, prompt, init_image, strength, 
            height, width, num_inference_steps, guidance_scale, seed
        ],
        outputs=output_image
    )

if __name__ == "__main__":
    # Pull dynamic hosting port injected by Google Cloud Run
    port = int(os.getenv("PORT", 8080))
    demo.launch(server_name="0.0.0.0", server_port=port)
