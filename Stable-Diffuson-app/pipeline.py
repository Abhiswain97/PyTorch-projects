import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16 
)
pipe.to("cuda")

print(pipe("a unicorn").images[0])

