import os
import argparse
from typing import List


def generate_filename_suffix(args: argparse.Namespace) -> str:
    """
    Generate a suffix for filenames based on the provided arguments.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        str: A string representing the filename suffix.
    """
    suffix = ""

    if (
        hasattr(args, "replacement_lm_name_or_path")
        and args.replacement_lm_name_or_path
    ):
        suffix += "_lm-" + args.replacement_lm_name_or_path.split("/")[-1]
    if (
        hasattr(args, "pretrained_projector_name_or_path")
        and args.pretrained_projector_name_or_path
        and args.replace_projector
    ):
        suffix += "_pproj"
    if hasattr(args, "texts_qa") and args.texts_qa:
        suffix += "_txt-qa"
    if hasattr(args, "images_qa") and args.images_qa:
        suffix += "_img-qa"
    if hasattr(args, "chat_mode"):
        if args.chat_mode:
            suffix += "_chat"
        else:
            suffix += "_ntp"
    if hasattr(args, "question_instruction_type") and args.question_instruction_type:
        suffix += f"_qitype-{args.question_instruction_type}"
    if hasattr(args, "guide_text") and args.guide_text:
        suffix += "_gtxt"
    if hasattr(args, "continue_final_message") and args.continue_final_message:
        suffix += "_cfm"
    if hasattr(args, "downsample_size") and args.downsample_size:
        suffix += f"_ds{args.downsample_size}"
    if hasattr(args, "seed") and args.seed:
        suffix += f"_s{args.seed}"
    if hasattr(args, "batch_size") and args.batch_size:
        suffix += f"_batch{args.batch_size}"

    return suffix


def create_filename(args: argparse.Namespace) -> str:
    """
    Create a descriptive filename based on extraction parameters.
    """
    # Handle layer specification
    filename = "layer-"
    if isinstance(args.layer_index, int):
        if args.layer_index == -1:
            filename += "last"
        else:
            if args.layer_index > 0:
                filename += str(args.layer_index)
            else:
                filename += str(args.layer_index).replace("-", "minus")
    elif isinstance(args.layer_index, list):
        filename += f"{','.join(str(num) for num in args.layer_index)}"
    else:
        filename += str(args.layer_index)

    # Handle token index or mean representation in the filename
    filename += "_token-"
    if hasattr(args, "token_index") and args.token_index is not None:
        if args.token_index == -1:
            filename += "last"  # Use "last" for the conventional last token index
        else:
            if args.token_index > 0:
                filename += str(args.token_index)
            else:
                filename += str(args.token_index).replace("-", "minus")
    elif hasattr(args, "mean_over_tokens") and args.mean_over_tokens:
        filename += "mean"

    if hasattr(args, "tokens_mode") and args.tokens_mode:
        filename += f"{args.tokens_mode}"

    # Create suffix based on other parameters
    suffix = generate_filename_suffix(args)

    filename += f"{suffix}.safetensors"
    return filename


def setup_directories(
    representations_dir: str,
    model_name: str,
    dataset_name: str,
    split: str = "",
) -> str:
    """
    Create necessary directories for saving representations.

    Args:
        representations_dir: Base directory for all saved representations
        model_name: Name of the model being processed
        dataset_name: Dataset name or path
        split: Dataset split (e.g., "train", "test", "val")

    Returns:
        str: Path to the directory where files should be saved
    """
    # Clean up model name for folder naming
    model_folder = model_name.replace("/", "_")

    # Get dataset name from path
    dataset_name = dataset_name.split("/")[-1]

    # Create directories with model and split subfolder
    save_dir = os.path.join(representations_dir, model_folder, dataset_name, split)
    os.makedirs(save_dir, exist_ok=True)

    return save_dir


def create_filename_from_paths(
    path1: str, path2: str, args: argparse.Namespace
) -> List[str]:
    """
    Create a suffix for filenames based on the provided paths.
    """
    split_path1 = path1.split("representations", 1)
    split_path2 = path2.split("representations", 1)
    remainder1 = split_path1[1].split(".safetensors")[0].split("/")[1:]
    remainder2 = split_path2[1].split(".safetensors")[0].split("/")[1:]
    model_name1 = remainder1[0]
    model_name2 = remainder2[0]
    details = (
        "_".join(remainder1[1::])
        if len("_".join(remainder1[1::])) > len("_".join(remainder2[1::]))
        else "_".join(remainder2[1::])
    )
    details = (
        details.replace(model_name1, "").replace(model_name2, "").replace("lm-_", "")
    )
    suffix = f"{model_name1}_vs_{model_name2}_{details}"
    if args.measure == "neighborhood_overlap" and args.maxk is not None:
        suffix += f"_maxk-{args.maxk}"
    if args.measure == "svcca" and args.accept_rate is not None:
        suffix += f"_arate-{args.accept_rate}"
    if args.measure == "rbf_cka" and args.sigma is not None:
        suffix += f"_rbf-s{args.sigma}"
    if "downsample" not in suffix and args.downsample_size is not None:
        suffix += f"_reps-ds{args.downsample_size}"
    return suffix, model_name1, model_name2
