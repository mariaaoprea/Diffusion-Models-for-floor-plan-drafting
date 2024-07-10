from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("model/checkpoint-250", weight_name="pytorch_lora_weights.safetensors")
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

labels = ["BFMBM", "BFOBM", "BMMBM", "BMOBM", "SFMBM", "SFOBM", "SFOBF", "SFOSM", "SFOSF"]


for prompt, label in zip(prompts, labels):
    for i in range(1,11):
        image = pipeline(prompt).images[0]
        image.save(f"images/model/{label}_{i}.png")
