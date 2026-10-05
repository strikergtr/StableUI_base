import os
import random
import uuid
import sys
import torch
import numpy as np
import gradio as gr
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

# Constants & Paths
MODEL_DIR = "/content/StableUI_base"
SAVE_DIR = "/content/images"
MODEL_PATH = os.path.join(MODEL_DIR, "model_link.safetensors")
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1344

# Default SDXL Checkpoint URL
# หมายเหตุ: หากโมเดลต้องการ API Key ให้ต่อท้ายด้วย &token=YOUR_CIVITAI_API_KEY
MODEL = "https://civitai.com/api/download/models/128078?type=Model&format=SafeTensor&size=pruned&fp=fp16"

if len(sys.argv) > 1:
    MODEL = sys.argv[1]

# Create required directories
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Device & Precision setup
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Download model if not exists
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100_000_000:
    print("⏳ Downloading model checkpoint...")
    os.system(f'wget -c --content-disposition -O "{MODEL_PATH}" "{MODEL}"')

# Load Pipeline
print("⏳ Loading pipeline into memory...")
pipe = StableDiffusionXLPipeline.from_single_file(
    MODEL_PATH,
    torch_dtype=dtype,
    use_safetensors=True
)

if device == "cuda":
    # ลด VRAM usage บน Free T4 GPU
    pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
else:
    pipe.to(device)

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
print("\033[1;32mPipeline ready!\033[0m")

def infer(prompt, negative_prompt, seed, width, height, guidance_scale, num_inference_steps):
    if seed == -1:
        seed = random.randint(0, MAX_SEED)
        
    generator = torch.Generator(device="cpu").manual_seed(seed) # ใช้ CPU seed generator ป้องกัน GPU state desync
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=int(num_inference_steps),
        width=int(width),
        height=int(height),
        generator=generator,
    ).images[0]
    
    image_filename = f"{uuid.uuid4()}.png"
    image_path = os.path.join(SAVE_DIR, image_filename)
    image.save(image_path)
    
    return image

# UI setup
css = """
#col-container {
    margin: 0 auto;
    max-width: 680px;
}
footer {
    display: none !important;
}
"""

examples = [
    "a cinematic shot of an astronaut riding a horse on mars, highly detailed, 8k",
    "cute anime girl with cat ears sitting by a coffee shop window, rainy day, studio ghibli style",
    "a majestic lion wearing royal crown, portrait, fantasy concept art"
]

with gr.Blocks(css=css, theme='ParityError/Interstellar') as app:
    with gr.Column(elem_id="col-container"):
        gr.Markdown("""
        # 🎨 Stable Diffusion XL Workspace
        *Google Colab Session — ข้อมูลและรูปภาพจะถูกลบเมื่อปิด Runtime*
        """)

        with gr.Group():
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...", lines=2)
            run_button = gr.Button("🚀 Generate Image", variant='primary')
        
        result = gr.Image(label="Generated Result", interactive=False)
        
        with gr.Accordion("⚙️ Advanced Settings", open=False):
            negative_prompt = gr.Textbox(
                label="Negative prompt", 
                lines=2, 
                value='lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, mutilated, bad anatomy, deformed'
            )
            
            seed = gr.Slider(label="Seed (-1 for random)", minimum=-1, maximum=MAX_SEED, step=1, value=-1)
            
            with gr.Row():
                width = gr.Slider(label="Width", minimum=512, maximum=MAX_IMAGE_SIZE, step=64, value=1024)
                height = gr.Slider(label="Height", minimum=512, maximum=MAX_IMAGE_SIZE, step=64, value=1024)
            
            with gr.Row():
                guidance_scale = gr.Slider(label="CFG / Guidance scale", minimum=1.0, maximum=15.0, step=0.5, value=7.0)
                num_inference_steps = gr.Slider(label="Sampling Steps", minimum=10, maximum=50, step=1, value=25)

        gr.Examples(examples=examples, inputs=[prompt])
    
    run_button.click(
        fn=infer,
        inputs=[prompt, negative_prompt, seed, width, height, guidance_scale, num_inference_steps],
        outputs=result,
    )

if __name__ == "__main__":
    app.queue().launch(share=True, debug=True)
