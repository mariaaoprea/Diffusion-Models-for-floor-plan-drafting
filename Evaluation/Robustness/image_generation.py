from diffusers import AutoPipelineForText2Image
import torch

# Load the pre-trained AutoPipelineForText2Image model
pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")

# Load the LoRA weights for the model
pipeline.load_lora_weights("Checkpoint_L1/checkpoint-250", weight_name="pytorch_lora_weights.safetensors")

# Define a list of prompts for generating floor plan images
prompts = [
    "Floor plan of a small apartment, few rooms, one bathroom, small kitchen, few windows",
    "Floor plan of a big apartment, few rooms, multiple bathrooms, small kitchen, few windows",
    "A compact apartment layout, limited rooms, one washroom, a kitchenette, limited fenestration",
    "A floor plan of a small apartment that includes a small kitchen and one bathroom, with few rooms and few windows",
    "Floor plan of a small apartment, few rooms, one bathroom, small kitchen, few windows, a balcony",
    "Floor plan of a small apartment, few rooms, one bathroom, a kitchen with an island, few windows",
    "Floor plan of a small apartment, few rooms, one bathroom, a big kitchen, few windows, and a central hallway connecting all rooms",
    "Layout of an artist’s loft with an open plan living area, a mezzanine level bedroom, a bathroom to the left of the kitchenette, a small studio space next to a big balcony"
]

# Define a list of labels for the generated images
labels = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]

# Generate and save the floor plan images for each prompt and label
for prompt, label in zip(prompts, labels):
    for i in range(1, 11):
        # Generate the image using the pipeline
        image = pipeline(prompt).images[0]
        
        # Save the generated image with the corresponding label and index
        image.save(f"Evaluation/Robustness/images/L1_rank4/{label}_{i}.png")


# Load the pre-trained AutoPipelineForText2Image model
pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")

# Load the LoRA weights for the model
pipeline.load_lora_weights("Checkpoints_L1_r6/checkpoint-250", weight_name="pytorch_lora_weights.safetensors")

# Generate and save the floor plan images for each prompt and label
for prompt, label in zip(prompts, labels):
    for i in range(1, 11):
        # Generate the image using the pipeline
        image = pipeline(prompt).images[0]
        
        # Save the generated image with the corresponding label and index
        image.save(f"Evaluation/Robustness/images/L1_rank6/{label}_{i}.png")