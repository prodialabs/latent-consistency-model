#!/usr/bin/env python
from __future__ import annotations

import os
import random
import time

import numpy as np
import PIL.Image
import torch

from diffusers import DiffusionPipeline
import torch

import os
import torch
from tqdm import tqdm
from safetensors.torch import load_file
import gradio_user_history as gr_user_history

from concurrent.futures import ThreadPoolExecutor
import uuid
import cv2

from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import Optional, Union

import base64
from io import BytesIO

DESCRIPTION = '''# Latent Consistency Model
Distilled from [Dreamshaper v7](https://huggingface.co/Lykon/dreamshaper-7) fine-tune of [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) with only 4,000 training iterations (~32 A100 GPU Hours). [Project page](https://latent-consistency-models.github.io)
'''
if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES") == "1"
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "768"))
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE") == "1"
DTYPE = torch.float32  # torch.float16 works as well, but pictures seem to be a bit worse

pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7", custom_pipeline="latent_consistency_txt2img", custom_revision="main")
pipe.to(torch_device="cuda", torch_dtype=DTYPE)


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def save_image(img, profile: gr.OAuthProfile | None, metadata: dict):
    unique_name = str(uuid.uuid4()) + '.png'
    img.save(unique_name)
    gr_user_history.save_image(label=metadata["prompt"], image=img, profile=profile, metadata=metadata)
    return unique_name

def save_images(image_array, profile: gr.OAuthProfile | None, metadata: dict):
    paths = []
    with ThreadPoolExecutor() as executor:
        paths = list(executor.map(save_image, image_array, [profile]*len(image_array), [metadata]*len(image_array)))
    return paths

paths = []

def images_to_base64(images: List[Image.Image], format: str = "PNG") -> List[str]:
    """
    Convert a list of PIL images to a list of base64 encoded strings.

    Args:
    - images (List[Image.Image]): List of PIL images.
    - format (str): The format to save the image in. Default is "PNG".

    Returns:
    - List[str]: List of base64 encoded image data strings.
    """
    encoded_images = []

    for image in images:
        buffered = BytesIO()
        image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        encoded_images.append(img_str)

    return encoded_images

def generate(
    prompt: str,
    seed: int = 0,
    width: int = 512,
    height: int = 512,
    guidance_scale: float = 8.0,
    num_inference_steps: int = 4,
    num_images: int = 4,
    randomize_seed: bool = False,
    # progress = gr.Progress(track_tqdm=True),
    # profile: gr.OAuthProfile | None = None,
) -> PIL.Image.Image:
    seed = randomize_seed_fn(seed, randomize_seed)
    torch.manual_seed(seed)
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

    return images_to_base64(result)
    
examples = [
    "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
    "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
]

app = FastAPI()

# Define the Pydantic model for the request body
class ImageRequest(BaseModel):
    prompt: str
    seed: int = 0
    width: int = 512
    height: int = 512
    guidance_scale: float = 8.0
    num_inference_steps: int = 4
    num_images: int = 4

@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    return  generate(
        request.prompt,
        request.seed,
        request.width,
        request.height,
        request.guidance_scale,
        request.num_inference_steps,
        request.num_images,
        True
    )

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4000)
