from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("output/checkpoint-15000", weight_name="pytorch_lora_weights.safetensors")
image = pipeline("a floorplan of a small apartment,many rooms, small kitchen, many windows, one bathroom").images[0]
image.save("images/fp7.jpg")
#image = pipeline("a pokemon").images[0]
#image.save("images/new_with2.jpg")