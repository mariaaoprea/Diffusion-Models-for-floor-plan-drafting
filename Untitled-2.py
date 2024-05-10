from diffusers import DiffusionPipeline
import torch
pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
pipeline.to("cuda")
image = pipeline("An image of a squirrel in Picasso style").images[0]
#image.save("image_of_squirrel_painting.png")