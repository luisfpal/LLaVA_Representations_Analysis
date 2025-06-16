import os
import argparse


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
        suffix += "_lm-" + args.replacement_lm_name_or_path.replace("/", "-")
    if hasattr(args, "texts_qa") and args.texts_qa:
        suffix += "_texts-qa"
    if hasattr(args, "images_qa") and args.images_qa:
        suffix += "_images-qa"
    if hasattr(args, "chat_mode"):
        if args.chat_mode:
            suffix += "_chat-format"
        else:
            suffix += "_ntp_format"
    if hasattr(args, "question_instruction_type") and args.question_instruction_type:
        suffix += f"_qinst-type-{args.question_instruction_type}"
    if hasattr(args, "guide_text") and args.guide_text:
        suffix += "_guide-text"
    if hasattr(args, "continue_final_message"):
        if args.continue_final_message:
            suffix += "_continue-fm"
    if hasattr(args, "remove_images"):
        if args.remove_images:
            suffix += "_captions-context"
        else:
            suffix += "_images-context"
    if hasattr(args, "downsample_size") and args.downsample_size:
        suffix += f"_downsample-{args.downsample_size}"
    if hasattr(args, "seed") and args.seed:
        suffix += f"_seed-{args.seed}"
    if hasattr(args, "batch_size") and args.batch_size:
        suffix += f"_batch-{args.batch_size}"

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
        filename += f"{','.join(str(num) for num in args.layer_index)}-"
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

    if hasattr(args, "tokens_mode"):
        filename += args.tokens_mode

    # Create suffix based on other parameters
    suffix = generate_filename_suffix(args)

    filename += f"{suffix}.safetensors"
    return filename


def setup_directories(
    representations_dir: str, model_name: str, dataset_name: str, split: str
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
