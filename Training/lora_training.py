import logging
import os
from pathlib import Path
import datasets
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import CLIPTextModel
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
import wandb
from arguments import parse_args
from preprocessing import preprocess_data

import torch.nn.functional as F
import torch.utils.checkpoint

#This is the training script for the LoRA model. 
#It is based on the training script from the huggingface diffusers repository: https://github.com/huggingface/diffusers.git.
#The train_text_to_image.py script was then split into all python files that can be found in the training folder 
#and then modified to the needs of this thesis

# Imported files

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.28.0.dev0")

def main():
    """
    Main function to run the training process.
    """
    # Parse the arguments
    args = parse_args()
    # Initialize the logger
    logger = get_logger(__name__, log_level="INFO")
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    # Initialize the accelerator
    accelerator = Accelerator(
        log_with=args.report_to,
        project_config=accelerator_project_config
    )
    # Load the preprocessed dataset
    train_dataloader = preprocess_data(accelerator)

    # Initialize wandb project    
    wandb.init(project="stable-diffusion-lora")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    #logging settings
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    # Set the training seed for reproductible training
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    # Set the weight dtype
    weight_dtype = torch.float32

    # Freeze the unet parameters before adding adapters
    for param in unet.parameters():
        param.requires_grad_(False)
    # Define the LoraConfig
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Add adapter 
    unet.add_adapter(unet_lora_config)
    # Unfreeze the lora layers
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    # Unwrap the model if it is a compiled module
    def unwrap_model(model):
        """
        Unwraps a model by removing any accelerator-specific wrappers.

        Args:
            model: The model to unwrap.

        Returns:
            The unwrapped model.

        """
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Initialize the optimizer - Use AdamW
    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Calculate the number of update steps per epoch
    num_update_steps_per_epoch = len(train_dataloader)
    train_steps = args.num_train_epochs * num_update_steps_per_epoch

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=train_steps
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Initialize the trackers
    if accelerator.is_main_process:
        accelerator.init_trackers("floorplan-LoRA", config=vars(args))

    # Initialize the training loop
    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {train_steps}")
    global_step = 0
    first_epoch = 1
    previous_epochs = 0
    
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            first_epoch = int(path.split("-")[1])+1
            previous_epochs = int(path.split("-")[1])

    # Initialize the progress bar
    progress_bar = tqdm(
            range(0, train_steps),
            initial=0,
            desc="Steps"
        )

    #generate images before training
    logger.info(
    f"Running first inference... \n Generating {args.num_validation_images} images with prompt:"
    f" {args.validation_prompt}."
    )
    # create pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unwrap_model(unet),
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
    images = []
    
    autocast_ctx = torch.autocast(accelerator.device.type)

    # Generate images
    with autocast_ctx:
        for _ in range(args.num_validation_images):
            images.append(
                pipeline(args.validation_prompt, num_inference_steps=30, generator=generator).images[0]
            )

    accelerator.log(
        {
            "validation": [
                wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                for i, image in enumerate(images)
            ]
        }
    )

    del pipeline

    # Training loop
    for epoch in range(first_epoch, args.num_train_epochs + 1 + previous_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Generate noise for the forward diffusion process
                noise = torch.randn_like(latents)

                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
                timesteps = timesteps.long()

                # Forward diffusion process
                # Add noise to the latents according to the noise magnitude at each timestep
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]

                if args.loss_function == "MSE":
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                elif args.loss_function == "L1":
                    loss = F.l1_loss(model_pred.float(), target.float(), reduction="mean")
                elif args.loss_function == "SNR":
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({
                    "train_loss": loss,
                }, step=global_step)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)                

        #evaluation mode  
        unet.eval()


        # Save the model checkpoint once per epoch
        if epoch % args.checkpointing_frequency == 0:
            if accelerator.is_main_process:
                save_path = os.path.join(args.output_dir, f"checkpoint-{epoch}")
                accelerator.save_state(save_path)

                unwrapped_unet = unwrap_model(unet)
                unet_lora_state_dict = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(unwrapped_unet)
                )

                StableDiffusionPipeline.save_lora_weights(
                    save_directory=save_path,
                    unet_lora_layers=unet_lora_state_dict,
                    safe_serialization=True,
                )

            logger.info(f"Saved state to {save_path}")

        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_prompt}."
                )
                # create pipeline
                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=unwrap_model(unet),
                    torch_dtype=weight_dtype,
                )
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = torch.Generator(device=accelerator.device)
                if args.seed is not None:
                    generator = generator.manual_seed(args.seed)
                images = []
                
                autocast_ctx = torch.autocast(accelerator.device.type)

                # Generate images
                with autocast_ctx:
                    for _ in range(args.num_validation_images):
                        images.append(
                            pipeline(args.validation_prompt, num_inference_steps=30, generator=generator).images[0]
                        )

                accelerator.log(
                    {
                        "validation": [
                            wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                            for i, image in enumerate(images)
                        ]
                    }
                )

                del pipeline
                torch.cuda.empty_cache()

    # Wait for all processes to be done
    accelerator.wait_for_everyone()
    # Close the progress bar
    accelerator.end_training()

if __name__ == "__main__":
    main()
