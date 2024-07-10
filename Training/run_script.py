import subprocess

def main():
    """
    Main function to execute the training script.
    """
    
    # Build the command
    command = [
        "accelerate", "launch", 
        "lora_training.py",
        "--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5",
        "--train_data_dir=dataset",
        "--output_dir=output_rank8",
        "--cache_dir=cache",
        "--random_flip",
        "--train_batch_size=4",
        "--num_train_epochs=50",
        "--learning_rate=1e-04",
        "--validation_epochs=50",
        "--lr_scheduler=cosine",
        "--lr_warmup_steps=0",
        "--validation_prompt='Floor plan of a small apartment, few rooms, one bathroom, small kitchen, few windows'",
        "--seed=1337",
        "--rank=8",
        "--num_validation_images=4",
        "--checkpointing_frequency=50",
        "--resume_from_checkpoint=output_rank8/checkpoint-200",
        "--loss_function=L1"
    ]

    # Execute the command
    subprocess.run(command)

if __name__ == "__main__":
    main()