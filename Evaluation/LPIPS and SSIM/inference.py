from diffusers import AutoPipelineForText2Image
import torch

# Load the pre-trained AutoPipelineForText2Image model
pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")

# Load the LoRA weights for the model
pipeline.load_lora_weights("Checkpoints_SNR/checkpoint-250", weight_name="pytorch_lora_weights.safetensors")

# Define a list of prompts for generating floor plan images
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

# Define a list of labels for the generated images
labels = ["BFMBM", "BFOBM", "BMMBM", "BMOBM", "SFMBM", "SFOBM", "SFOBF", "SFOSM", "SFOSF"]

# Generate and save the floor plan images for each prompt and label
for prompt, label in zip(prompts, labels):
    for i in range(1, 11):
        # Generate the image using the pipeline
        image = pipeline(prompt).images[0]
        
        # Save the generated image with the corresponding label and index
        image.save(f"Evaluation/LPIPS and SSIM/images/SNR/{label}_{i}.png")
