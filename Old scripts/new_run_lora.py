import subprocess
import os

def main():
    # Set environment variables
    os.environ["MODEL_NAME"] = "runwayml/stable-diffusion-v1-5"
    os.environ["OUTPUT_DIR"] = "output_3"
    os.environ["HUB_MODEL_ID"] = "pokemon-lora"
    os.environ["DATASET_NAME"] = "diffusers/pokemon-gpt4-captions"
    os.environ["Directory"] = "dataset"
    


    # Build the command
    command = [
        "accelerate", "launch", 
        "new_lora.py",
        f"--pretrained_model_name_or_path={os.environ['MODEL_NAME']}",
        f"--train_data_dir={os.environ['Directory']}",
        #f"--dataset_name={os.environ['DATASET_NAME']}",
        "--dataloader_num_workers=8",
        "--resolution=512",
        "--center_crop",
        "--random_flip",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=4",
        "--max_train_steps=10",
        "--learning_rate=1e-04",
        "--max_grad_norm=1",
        "--lr_scheduler=cosine",
        "--lr_warmup_steps=0",
        f"--output_dir={os.environ['OUTPUT_DIR']}",
        f"--hub_model_id={os.environ['HUB_MODEL_ID']}",
        "--checkpointing_steps=500",
        "--validation_prompt='floorplan'",
        "--seed=1337",
        "--use_8bit_adam"

    ]

    # Execute the command
    subprocess.run(command)

if __name__ == "__main__":
    main()
