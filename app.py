#!/usr/bin/env python
from __future__ import annotations

import os
import random
import time

import gradio as gr
import numpy as np
import PIL.Image
import torch
from lcm_pipeline import LatentConsistencyModelPipeline
from lcm_scheduler import LCMScheduler

from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPTokenizer, CLIPTextModel, CLIPImageProcessor

import os
import torch
from tqdm import tqdm
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

DESCRIPTION = "# Latent Consistency Model"
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "768"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE") == "1"
DTYPE = torch.float32  # torch.float16 works as well, but pictures seem to be a bit worse

model_id = "digiplay/DreamShaper_7"


# Initalize Diffusers Model:
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
config = UNet2DConditionModel.load_config(model_id, subfolder="unet")
config["time_cond_proj_dim"] = 256

unet = UNet2DConditionModel.from_config(config)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(model_id, subfolder="safety_checker")
feature_extractor = CLIPImageProcessor.from_pretrained(model_id, subfolder="feature_extractor")

# Initalize Scheduler:
scheduler = LCMScheduler(beta_start=0.00085, beta_end=0.0120, beta_schedule="scaled_linear", prediction_type="epsilon")

HF_TOKEN = os.environ.get("HF_TOKEN", None)

if torch.cuda.is_available():
    # Replace the unet with LCM:
    # lcm_unet_ckpt = hf_hub_download("SimianLuo/LCM_Dreamshaper_v7", filename="LCM_Dreamshaper_v7_4k.safetensors", token=HF_TOKEN)
    lcm_unet_ckpt = "./LCM_Dreamshaper_v7_4k.safetensors"
    ckpt = load_file(lcm_unet_ckpt)
    m, u = unet.load_state_dict(ckpt, strict=False)
    if len(m) > 0:
        print("missing keys:")
        print(m)
    if len(u) > 0:
        print("unexpected keys:")
        print(u)


    # LCM Pipeline:
    pipe = LatentConsistencyModelPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler, safety_checker=safety_checker, feature_extractor=feature_extractor)
    pipe = pipe.to(torch_device="cuda", torch_dtype=DTYPE)

    if USE_TORCH_COMPILE:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def generate(
    prompt: str,
    seed: int = 0,
    width: int = 512,
    height: int = 512,
    guidance_scale: float = 8.0,
    num_inference_steps: int = 4,
    num_images: int = 4,
) -> PIL.Image.Image:
    torch.manual_seed(seed)

    # if width > 512 or height > 512:
    #     num_images = 2
    start_time = time.time()
    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images,
        lcm_origin_steps=50,
        output_type="pil",
    ).images
    print(time.time() - start_time)
    return result

examples = [
    "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
]

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=os.getenv("SHOW_DUPLICATE_BUTTON") == "1",
    )
    with gr.Group():
        with gr.Row():
            prompt = gr.Text(
                label="Prompt",
                show_label=False,
                max_lines=1,
                placeholder="Enter your prompt",
                container=False,
            )
            run_button = gr.Button("Run", scale=0)
        result = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery", grid=[2]
        )
    with gr.Accordion("Advanced options", open=False):
        seed = gr.Slider(
            label="Seed",
            minimum=0,
            maximum=MAX_SEED,
            step=1,
            value=0,
        )
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        with gr.Row():
            width = gr.Slider(
                label="Width",
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=512,
            )
            height = gr.Slider(
                label="Height",
                minimum=256,
                maximum=MAX_IMAGE_SIZE,
                step=32,
                value=512,
            )
        with gr.Row():
            guidance_scale = gr.Slider(
                label="Guidance scale for base",
                minimum=2,
                maximum=14,
                step=0.1,
                value=8.0,
            )
            num_inference_steps = gr.Slider(
                label="Number of inference steps for base",
                minimum=1,
                maximum=8,
                step=1,
                value=4,
            )
        # with gr.Row():
        #     num_images = gr.Slider(
        #         label="Number of images"
        #         minimum=1,
        #         maximum=8,
        #         step=1,
        #         value=4,
        #     )

    gr.Examples(
        examples=examples,
        inputs=prompt,
        outputs=result,
        fn=generate,
        cache_examples=CACHE_EXAMPLES,
    )

    gr.on(
        triggers=[
            prompt.submit,
            run_button.click,
        ],
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=seed,
        queue=False,
        api_name=False,
    ).then(
        fn=generate,
        inputs=[
            prompt,
            seed,
            width,
            height,
            guidance_scale,
            num_inference_steps,
        ],
        outputs=result,
        api_name="run",
    )

if __name__ == "__main__":
    # demo.queue(max_size=20).launch()
    demo.launch()
