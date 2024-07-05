from diffusers import StableDiffusionPipeline
import torch
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import numpy as np

# Function to calculate CLIP score
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(images, prompts):
    images_int = (images * 255).astype("uint8")
    clip_score = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score), 4)

# Function to generate images in batches
def generate_images_in_batches(prompts, batch_size):
    all_images = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        torch.cuda.empty_cache()  # Clear CUDA cache
        images = sd_pipeline(batch_prompts, num_images_per_prompt=1, output_type="np").images
        all_images.append(images)
    return np.concatenate(all_images, axis=0)

# Load the model
model_ckpt = "runwayml/stable-diffusion-v1-5"
sd_pipeline = StableDiffusionPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16).to("cuda")

# Define prompts   #23 prompts with image, 32 in total, 9 over 10 images
prompts = [
    "Floor plan of a big apartment, few rooms, multiple bathrooms, big kitchen, many windows", #14
    "Floor plan of a big apartment, few rooms, multiple bathrooms, big kitchen, few windows", #2
    "Floor plan of a big apartment, few rooms, multiple bathrooms, small kitchen, many windows",#1
    "Floor plan of a big apartment, few rooms, multiple bathrooms, small kitchen, few windows", # 0
    "Floor plan of a big apartment, few rooms, one bathroom, big kitchen, many windows", #11
    "Floor plan of a big apartment, few rooms, one bathroom, big kitchen, few windows", #6
    "Floor plan of a big apartment, few rooms, one bathroom, small kitchen, many windows", #3
    "Floor plan of a big apartment, few rooms, one bathroom, small kitchen, few windows", #2   ######big+few=39
    "Floor plan of a big apartment, many rooms, multiple bathrooms, big kitchen, many windows", #36
    "Floor plan of a big apartment, many rooms, multiple bathrooms, big kitchen, few windows", #3
    "Floor plan of a big apartment, many rooms, multiple bathrooms, small kitchen, many windows", #7
    "Floor plan of a big apartment, many rooms, multiple bathrooms, small kitchen, few windows", #0
    "Floor plan of a big apartment, many rooms, one bathroom, big kitchen, many windows", #11
    "Floor plan of a big apartment, many rooms, one bathroom, big kitchen, few windows", #3
    "Floor plan of a big apartment, many rooms, one bathroom, small kitchen, many windows", #3
    "Floor plan of a big apartment, many rooms, one bathroom, small kitchen, few windows", #0   #####big#+many=63  ##big=102
    "Floor plan of a small apartment, few rooms, multiple bathrooms, big kitchen, many windows", #14
    "Floor plan of a small apartment, few rooms, multiple bathrooms, big kitchen, few windows", #3
    "Floor plan of a small apartment, few rooms, multiple bathrooms, small kitchen, many windows", #4
    "Floor plan of a small apartment, few rooms, multiple bathrooms, small kitchen, few windows", #0
    "Floor plan of a small apartment, few rooms, one bathroom, big kitchen, many windows", #41
    "Floor plan of a small apartment, few rooms, one bathroom, big kitchen, few windows", #38
    "Floor plan of a small apartment, few rooms, one bathroom, small kitchen, many windows", #19
    "Floor plan of a small apartment, few rooms, one bathroom, small kitchen, few windows", #50  ####small+few=169
    "Floor plan of a small apartment, many rooms, multiple bathrooms, big kitchen, many windows", #0
    "Floor plan of a small apartment, many rooms, multiple bathrooms, big kitchen, few windows", #0
    "Floor plan of a small apartment, many rooms, multiple bathrooms, small kitchen, many windows", #1
    "Floor plan of a small apartment, many rooms, multiple bathrooms, small kitchen, few windows", #0
    "Floor plan of a small apartment, many rooms, one bathroom, big kitchen, many windows", #5
    "Floor plan of a small apartment, many rooms, one bathroom, big kitchen, few windows", #0
    "Floor plan of a small apartment, many rooms, one bathroom, small kitchen, many windows", #1
    "Floor plan of a small apartment, many rooms, one bathroom, small kitchen, few windows" #0  ####small+many=7  ##small=176
]

# Generate images in batches
batch_size = 8  # Adjust batch size as needed
images = generate_images_in_batches(prompts, batch_size)

# Calculate CLIP score
sd_clip_score = calculate_clip_score(images, prompts)
print(f"CLIP score before: {sd_clip_score}")
sd_pipeline.load_lora_weights("output/checkpoint-15000", weight_name="pytorch_lora_weights.safetensors")
images = generate_images_in_batches(prompts, batch_size)
sd_clip_score = calculate_clip_score(images, prompts)
print(f"CLIP score after: {sd_clip_score}")