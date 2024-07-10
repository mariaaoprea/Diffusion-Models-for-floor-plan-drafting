from datasets import load_dataset
import os
import torch
from transformers import CLIPTokenizer
from torchvision import transforms
from Training.arguments import parse_args

args = parse_args()
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
    captions = []
    for caption in examples[caption_column]:
        captions.append(caption)
    inputs = tokenizer(
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

train_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["input_ids"] = tokenize_captions(examples)
    return examples

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}

def preprocess_data(accelerator):
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



