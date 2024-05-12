import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from new_args import parse_args
from pathlib import Path


args = parse_args()
logging_dir = Path(args.output_dir, args.logging_dir)
accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
    log_with=args.report_to,
    project_config=accelerator_project_config,
)
if torch.backends.mps.is_available():
    print("gg")
else:
    print("no")