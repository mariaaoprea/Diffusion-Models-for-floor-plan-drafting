from datasets import load_dataset
import os
import torch
from transformers import CLIPTokenizer
from torchvision import transforms
from arguments import parse_args

# Parse command line arguments
args = parse_args()

# Initialize the tokenizer
tokenizer = CLIPTokenizer.from_pretrained(
    args.pretrained_model_name_or_path, subfolder="tokenizer"
)

# Load dataset
data_files = {}
data_files["train"] = os.path.join(args.train_data_dir, "**")
dataset = load_dataset(
    "imagefolder",
    data_files=data_files,
    cache_dir=args.cache_dir,
)

# Get the column names
column_names = dataset["train"].column_names
image_column = column_names[0]
caption_column = column_names[1]

# Tokenize input captions and transform the images
def tokenize_captions(examples, is_train=True):
    """
    Tokenizes input captions using the CLIPTokenizer.

    Args:
        examples (dict): Dictionary containing input examples.
        is_train (bool): Whether the examples are from the training set.

    Returns:
        dict: Tokenized input captions.
    """
    captions = []
    for caption in examples[caption_column]:
        captions.append(caption)
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

# Define the image transformations for training
train_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def preprocess_train(examples):
    """
    Preprocesses the training examples by converting images to RGB, applying transformations, and tokenizing captions.

    Args:
        examples (dict): Dictionary containing training examples.

    Returns:
        dict: Preprocessed training examples.
    """
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["input_ids"] = tokenize_captions(examples)
    return examples

def collate_fn(examples):
    """
    Collates the preprocessed examples into batches.

    Args:
        examples (list): List of preprocessed examples.

    Returns:
        dict: Batched examples.
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}

def preprocess_data(accelerator):
    """
    Preprocesses the data and creates the training dataloader.

    Args:
        accelerator: The accelerator to use for distributed training.

    Returns:
        torch.utils.data.DataLoader: Training dataloader.
    """
    with accelerator.main_process_first():
        dataset["train"] = dataset["train"].shuffle(seed=args.seed)
        train_dataset = dataset["train"].with_transform(preprocess_train)

    # Create the dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size
    )
    return train_dataloader
