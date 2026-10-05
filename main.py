import os
import random
import uuid
import sys
import torch
import numpy as np
import gradio as gr
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

# Constants & Paths
SAVE_DIR = "/content/images"
MODEL_PATH = '/content/StableUI_base/model_link.safetensors'

# รับ path ของโมเดลผ่าน argument ถ้ามีการส่งเข้ามา เช่น: python main.py /path/to/model.safetensors
if len(sys.argv) > 1:
    MODEL_PATH = sys.argv[1]

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1344

# สร้างโฟลเดอร์สำหรับเซฟภาพ
os.makedirs(SAVE_DIR, exist_ok=True)

# Device & Precision setup
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# ตรวจสอบว่าพบไฟล์โมเดลหรือไม่ก่อนโหลด
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\n[Error] ไม่พบไฟล์โมเดลที่: {MODEL_PATH}\n"
        "กรุณาตรวจสอบว่าก๊อปปี้ไฟล์จาก Google Drive มาวางถูกตำแหน่ง หรือระบุ Path ถูกต้องแล้วหรือยัง"
    )

print(f"⏳ กำลังโหลดโมเดลจาก: {MODEL_PATH}")

# โหลดโมเดลผ่าน Diffusers Single File Loader
pipe = StableDiffusionXLPipeline.from_single_file(
    MODEL_PATH,
    torch_dtype=dtype,
    use_safetensors=True
)

# จัดการ Memory / VRAM
if device == "cuda":
    pipe.enable_model_cpu_offload()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
else:
    pipe.to(device)

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
print("\033[1;32mModel ready!\033[0m")

def infer(prompt, negative_prompt, seed, width, height, guidance_scale, num_inference_steps):
    if seed == -1:
        seed = random.randint(0, MAX_SEED)
        
    generator = torch.Generator(device="cpu").manual_seed(seed)
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=float(guidance_scale),
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
    max-width: 600px;
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
        # Stable Diffusion XL
        รันด้วย Local Model จาก Google Drive
        """)

        with gr.Group():
            prompt = gr.Textbox(label="Prompt", show_label=False, lines=2, placeholder="Enter your prompt here...")
            run_button = gr.Button("🚀 Generate", variant='primary')
        
        result = gr.Image(label="Result", interactive=False)
        
        with gr.Accordion("⚙️️ Settings", open=False):
            negative_prompt = gr.Textbox(
                label="Negative prompt", 
                lines=2, 
                value='lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, bad anatomy, deformed'
            )
            
            seed = gr.Slider(label="Seed (-1 for random)", minimum=-1, maximum=MAX_SEED, step=1, value=-1)
            
            with gr.Row():
                width = gr.Slider(label="Width", minimum=512, maximum=MAX_IMAGE_SIZE, step=64, value=1024)
                height = gr.Slider(label="Height", minimum=512, maximum=MAX_IMAGE_SIZE, step=64, value=1024)
            
            with gr.Row():
                guidance_scale = gr.Slider(label="Guidance scale (CFG)", minimum=1.0, maximum=15.0, step=0.5, value=7.0)
                num_inference_steps = gr.Slider(label="Steps", minimum=10, maximum=50, step=1, value=25)

        gr.Examples(examples=examples, inputs=[prompt])
    
    run_button.click(
        fn=infer,
        inputs=[prompt, negative_prompt, seed, width, height, guidance_scale, num_inference_steps],
        outputs=result,
    )

if __name__ == "__main__":
    app.queue().launch(share=True, debug=True)
