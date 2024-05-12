from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("output_2/checkpoint-500", weight_name="pytorch_lora_weights.safetensors")
image = pipeline("a blue house").images[0]
image.save("images/new_with1.jpg")
image = pipeline("a red apple").images[0]
image.save("images/new_with2.jpg")