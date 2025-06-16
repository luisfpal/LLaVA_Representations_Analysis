import os
import torch
import argparse
from typing import Optional
from utils import (
    load_hf_model_and_processor_or_tokenizer,
    parse_layer_index,
    create_filename,
    replace_multimodal_lm,
    parse_question_instruction,
    seed_all,
    setup_directories,
    save_extracted_residual_stream_data,
    get_dataloader,
)
from src.extract_residual_stream import extract_residual_stream


def extract_and_save_residual_streams(
    model_name: str,
    text_model: bool,
    args: argparse.Namespace,
    replacement_lm_name_or_path: Optional[str] = None,
    replace_lm: bool = False,
) -> str:
    """
    Extract and save residual streams from a model for a dataset.

    Args:
        model_name: Name or path of the model
        text_model: Boolean indicating if the model is a text model
        args: Argument namespace containing various parameters

    Returns:
        str: Path to the saved file
    """
    model_for_representations_extraction = (
        replacement_lm_name_or_path if replace_lm else model_name
    )

    print(f"\n=== Processing model: {model_for_representations_extraction} ===")

    # Setup save directory with split subfolder
    save_dir = setup_directories(
        args.representations_dir,
        model_for_representations_extraction,
        args.dataset_name,
        args.split,
    )

    save_dir = os.path.join(save_dir, "layers_representations")
    os.makedirs(save_dir, exist_ok=True)
    save_filename = create_filename(args)
    save_path = os.path.join(save_dir, save_filename)

    # Check if file already exists to avoid duplicate work
    if os.path.exists(save_path):
        print(f"File already exists at {save_path}. Skipping extraction.")
        return save_path

    try:
        # Load model and processor
        print("Loading model and processor...")
        model, processor = load_hf_model_and_processor_or_tokenizer(
            model_name_or_path=model_name,
            cache_dir=args.model_cache_dir,
            device_map="cuda:0",
            dtype=torch.float16,
            text_model=text_model,
        )

        # Set the model to evaluation mode
        model.eval()

        # Replace LLaVA language model
        if replacement_lm_name_or_path is not None and replace_lm:
            model = replace_multimodal_lm(
                multimodal_model=model,
                replacement_lm_name_or_path=replacement_lm_name_or_path,
                cache_dir=args.model_cache_dir,
            )

        # Get dataloader
        print("Preparing dataloader...")
        dataloader = get_dataloader(
            dataset_path_or_name=args.dataset_name,
            cache_dir=args.dataset_cache_dir,
            split=args.split,
            texts_qa=args.texts_qa,
            images_qa=args.images_qa,
            question_instruction_type=args.question_instruction_type,
            remove_images=args.remove_images,
            downsample_size=args.downsample_size,
            seed=args.seed,
        )

        # Extract residual streams
        residual_data = extract_residual_stream(
            model=model,
            processor=processor,
            dataloader=dataloader,
            args=args,
            text_model=text_model,
        )

        # Save the extracted data
        save_extracted_residual_stream_data(save_path, residual_data)

        # Free up memory
        del model, processor, residual_data
        torch.cuda.empty_cache()

        return save_path

    except Exception as e:
        print(f"Error during extraction: {str(e)}")
        raise


def main():
    """Main function to parse arguments and extract LLMs layers representations for multiple models."""
    parser = argparse.ArgumentParser(
        description="LLMs layers representations from multiple models"
    )
    parser.add_argument("--representations-dir", type=str, required=True)
    parser.add_argument("--model-cache-dir", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--dataset-cache-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--layer-index", type=parse_layer_index, default=-1)
    parser.add_argument("--token-index", type=int, default=-1)
    parser.add_argument("--mean-over-tokens", action="store_true", default=False)
    parser.add_argument("--texts_qa", action="store_true", default=False)
    parser.add_argument("--images_qa", action="store_true", default=False)
    parser.add_argument("--chat-mode", action="store_true", default=False)
    parser.add_argument("--mm-name-or-path", type=str, required=True)
    parser.add_argument("--lm-name-or-path", type=str, required=True)
    parser.add_argument(
        "--question-instruction-type", type=parse_question_instruction, default=None
    )
    parser.add_argument("--continue-final-message", action="store_true", default=False)
    parser.add_argument("--guide-text", type=str, default="")
    parser.add_argument("--remove-images", action="store_true", default=False)
    parser.add_argument("--downsample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Set the random seed for reproducibility
    seed_all(args.seed)

    # Expand user directory paths
    representations_dir = os.path.expanduser(args.representations_dir)

    # Make sure base save directory exists
    os.makedirs(representations_dir, exist_ok=True)

    # Setup
    parameters = [
        {"model_name": args.mm_name_or_path, "replace_lm": False},
        {"model_name": args.lm_name_or_path, "replace_lm": True},
    ]

    # Ensure only mean_over_tokens or token_index is set
    if args.mean_over_tokens:
        args.token_index = None

    # Extract and save for each model
    saved_paths = {}
    for param in parameters:
        model_for_representations_extraction = param["model_name"]
        try:
            save_path = extract_and_save_residual_streams(
                model_name=args.mm_name_or_path,
                text_model=False,
                args=args,
                replacement_lm_name_or_path=args.lm_name_or_path,
                replace_lm=param["replace_lm"],
            )

            saved_paths[model_for_representations_extraction] = save_path
            print(f"✅ Completed extraction for {model_for_representations_extraction}")

        except Exception as e:
            print(
                f"❌ Failed extraction for {model_for_representations_extraction}: {str(e)}"
            )

    # Print summary of saved paths
    print("\n=== Extraction Complete! ===")
    print("Saved representations:")
    for model_name, path in saved_paths.items():
        print(f"  {model_name}: {path}")


if __name__ == "__main__":
    main()
