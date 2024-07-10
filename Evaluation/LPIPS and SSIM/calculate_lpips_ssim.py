import lpips
import torch
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
import pandas as pd

# Function to load an image for LPIPS and convert to tensor
def load_image_for_lpips(filepath):
    """
    Load an image for LPIPS and convert it to a tensor.

    Args:
        filepath (str): The path to the image file.

    Returns:
        torch.Tensor: The image tensor.
    """
    img = Image.open(filepath).convert('RGB')  # Ensures image is in RGB format
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
    return img

# Function to load an image for SSIM and convert to RGB numpy array
def load_image_for_ssim(filepath):
    """
    Load an image for SSIM and convert it to a RGB numpy array.

    Args:
        filepath (str): The path to the image file.

    Returns:
        numpy.ndarray: The RGB image array.
    """
    img = Image.open(filepath).convert('RGB')  # Ensures image is in RGB format
    img = np.array(img).astype(np.float32) / 255.0
    return img

# Load LPIPS model
loss_fn = lpips.LPIPS(net='alex')

models = ["L1_r8, L1_r6, L1, MSE, SNR"]
labels = ["BFMBM", "BFOBM", "BMMBM", "BMOBM", "SFMBM", "SFOBM", "SFOBF", "SFOSM", "SFOSF"]

results = {model: [] for model in models}
mean_values = []

for model in models:
    for label in labels:
        lpips_values = []
        ssim_values = []
        for i in range(1, 11):
            target_lpips = load_image_for_lpips(f"images/Reference/{label}.png")
            img_lpips = load_image_for_lpips(f"images/{model}/{label}_{i}.png")
            target_ssim = load_image_for_ssim(f"images/Reference/{label}.png")
            img_ssim = load_image_for_ssim(f"images/{model}/{label}_{i}.png")
            
            lpips_value = loss_fn(target_lpips, img_lpips).item()
            ssim_value = ssim(target_ssim, img_ssim, data_range=1.0, channel_axis=-1)
            
            lpips_values.append(lpips_value)
            ssim_values.append(ssim_value)
        
        mean_lpips = np.mean(lpips_values)
        mean_ssim = np.mean(ssim_values)
        
        results[model].append({
            "Label": label,
            "Mean_LPIPS": mean_lpips,
            "Mean_SSIM": mean_ssim
        })

        mean_values.append({
            "Model": model,
            "Label": label,
            "Mean_LPIPS": mean_lpips,
            "Mean_SSIM": mean_ssim
        })

# Save individual model results to CSV
for model, data in results.items():
    df = pd.DataFrame(data)
    df.to_csv(f"LPIPS_SSIM/{model}_results.csv", index=False)

# Save the mean values to a CSV
mean_df = pd.DataFrame(mean_values)
mean_summary_df = mean_df.groupby("Model")[["Mean_LPIPS", "Mean_SSIM"]].mean().reset_index()
mean_summary_df.to_csv("LPIPS_SSIM/mean_values_L1_8.csv", index=False)

print("CSV files have been saved.")
