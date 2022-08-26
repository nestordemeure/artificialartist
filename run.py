import torch
from torch import autocast # will run the inference faster in half precision
from diffusers import StableDiffusionPipeline

from diffusionSimple import StableDiffusionPipelineSimple

# load model
# make sure you're logged in with `huggingface-cli login`
print("Loading the model...")
pipe = StableDiffusionPipelineSimple.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=True).to("cuda")

#prompt = "Tarot card depicting a caravan steered by a golden knight holding a cup of blood, by Caspar David Friedrich, matte painting trending on artstation HQ"
prompt = "tarot card depicting the magician holding his wand, by Vincent van Gogh, matte painting trending on artstation HQ"
height=512#+256
width=512
num_inference_steps=50
guidance_scale=7.5
download=True

# generate the images
with autocast("cuda"):
    # image here is in PIL format
    print("Running the model...")
    image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, height=height, width=width)["sample"][0]

# displays image
image.show()

# downloads the image
if download: 
    path = f"./output/{prompt}.png"
    image.save(path)

print("Done.")
