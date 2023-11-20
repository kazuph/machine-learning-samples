from diffusers import StableDiffusionControlNetPipeline, UNet2DConditionModel, ControlNetModel, LCMScheduler
import torch

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained("ckpt/SDHK", controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None,).to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")

import cv2
import numpy as np
from PIL import Image
import gradio as gr

def generate_canny(input_image, threshold1, threshold2):
    input_image = np.array(input_image)
    input_image = cv2.Canny(input_image, threshold1, threshold2)
    input_image = input_image[:, :, None]
    input_image = np.concatenate([input_image, input_image, input_image], axis=2)
    canny_image = Image.fromarray(input_image)
    return canny_image.resize((512, 768))

def generate(prompt, negative_prompt, controlnet_conditioning_scale, scale, num_inference_steps, guidance_scale, canny_image):
    image = pipe(prompt, negative_prompt=negative_prompt, image=canny_image, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, controlnet_conditioning_scale=float(controlnet_conditioning_scale), cross_attention_kwargs={"scale": float(scale)}).images[0]
    config = " | ".join([f"prompt: {prompt}", f"negative_prompt: {negative_prompt}", f"controlnet_conditioning_scale: {controlnet_conditioning_scale}", f"scale: {scale}", f"num_inference_steps: {num_inference_steps}", f"guidance_scale: {guidance_scale}"])
    return image.resize((512, 768)), config

with gr.Blocks(title=f"Realtime Latent Consistency Model") as demo:
    with gr.Box(scale=23):
      with gr.Row():
        with gr.Column():
            with gr.Row():
              prompt = gr.Textbox(show_label=False, value="1girl red dress")
            with gr.Row():
              negative_prompt = gr.Textbox(show_label=False, value="blurry")
        with gr.Column():
            with gr.Row():
              scale = gr.Slider(minimum=0, maximum=1, step=0.1, value=1, label="lora_scale")
              threshold1 = gr.Slider(minimum=0, maximum=500, step=1, value=100, label="threshold1")
              threshold2 = gr.Slider(minimum=0, maximum=500, step=1, value=200, label="threshold2")
            with gr.Row():
              num_inference_steps = gr.Slider(minimum=1, maximum=50, step=1, value=4, label="num_inference_steps")
              guidance_scale = gr.Slider(minimum=0, maximum=10, step=0.5, value=1, label="guidance_scale")
              controlnet_conditioning_scale = gr.Slider(minimum=0, maximum=1, step=0.1, value=0.5, label="controlnet_scale")
    with gr.Row():
        input_image = gr.Image(
            show_label=False,
            type="pil",
            tool="color-sketch",
            source="canvas",
            width=512,
            height=768,
            brush_radius=5.0,
            interactive=True,
        )
        canny_image = gr.Image(
            show_label=False,
            type="pil",
            tool="color-sketch",
            source="upload",
            interactive=True,
            width=512,
            height=768,
        )
        output_image = gr.Image(
            show_label=False,
            type="pil",
            tool="color-sketch",
            source="upload",
            interactive=False,
            width=512,
            height=768,
        )
    with gr.Row():
      config = gr.Label(show_label=False)

    input_image.change(fn=generate_canny, inputs=[input_image, threshold1, threshold2], outputs=[canny_image], show_progress=False)
    threshold1.change(fn=generate_canny, inputs=[input_image, threshold1, threshold2], outputs=[canny_image], show_progress=False)
    threshold2.change(fn=generate_canny, inputs=[input_image, threshold1, threshold2], outputs=[canny_image], show_progress=False)
    prompt.change(fn=generate, inputs=[prompt, negative_prompt, controlnet_conditioning_scale, scale, num_inference_steps, guidance_scale, canny_image], outputs=[output_image, config], show_progress=False)
    negative_prompt.change(fn=generate, inputs=[prompt, negative_prompt, controlnet_conditioning_scale, scale, num_inference_steps, guidance_scale, canny_image], outputs=[output_image, config], show_progress=False)
    controlnet_conditioning_scale.change(fn=generate, inputs=[prompt, negative_prompt, controlnet_conditioning_scale, scale, num_inference_steps, guidance_scale, canny_image], outputs=[output_image, config], show_progress=False)
    scale.change(fn=generate, inputs=[prompt, negative_prompt, controlnet_conditioning_scale, scale, num_inference_steps, guidance_scale, canny_image], outputs=[output_image, config], show_progress=False)
    num_inference_steps.change(fn=generate, inputs=[prompt, negative_prompt, controlnet_conditioning_scale, scale, num_inference_steps, guidance_scale, canny_image], outputs=[output_image, config], show_progress=False)
    guidance_scale.change(fn=generate, inputs=[prompt, negative_prompt, controlnet_conditioning_scale, scale, num_inference_steps, guidance_scale, canny_image], outputs=[output_image, config], show_progress=False)
    canny_image.change(fn=generate, inputs=[prompt, negative_prompt, controlnet_conditioning_scale, scale, num_inference_steps, guidance_scale, canny_image], outputs=[output_image, config], show_progress=False)
 
demo.launch(inline=False, share=True, debug=True, server_port=7860, server_name="0.0.0.0")
