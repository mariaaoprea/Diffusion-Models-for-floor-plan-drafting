from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("output_rank8/checkpoint-250", weight_name="pytorch_lora_weights.safetensors")
prompts = [
    "Floor plan of a big apartment, few rooms, multiple bathrooms, big kitchen, many windows",
    "Floor plan of a big apartment, few rooms, one bathroom, big kitchen, many windows",
    "Floor plan of a big apartment, many rooms, multiple bathrooms, big kitchen, many windows",
    "Floor plan of a big apartment, many rooms, one bathroom, big kitchen, many windows",
    "Floor plan of a small apartment, few rooms, multiple bathrooms, big kitchen, many windows",
    "Floor plan of a small apartment, few rooms, one bathroom, big kitchen, many windows",
    "Floor plan of a small apartment, few rooms, one bathroom, big kitchen, few windows",
    "Floor plan of a small apartment, few rooms, one bathroom, small kitchen, many windows",
    "Floor plan of a small apartment, few rooms, one bathroom, small kitchen, few windows"
]

labels = ["SFOSF"]

for prompt, label in zip(prompts, labels):
    for i in range(1,11):
        image = pipeline(prompt).images[0]
        image.save(f"images/L1_8/{label}_{i}.png")

#image = pipeline("a floor plan of a small apartment").images[0]
#image.save("images/fp7.jpg")
#image = pipeline("a pokemon").images[0]
#image.save("images/new_with2.jpg")