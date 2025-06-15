import os
import torch
import gc
from typing import Dict, Optional
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import traceback
from utils import (
    load_hf_model_and_processor_or_tokenizer,
    replace_multimodal_lm,
    seed_all,
    parse_layer_index,
    parse_question_instruction,
    format_prompts,
    setup_directories,
    create_filename,
    parse_tokens_mode,
    save_extracted_residual_stream_data,
    get_dataloader,
)
from src.extract_heads_representations import HeadProjectionTracer


def heads_representations_extractor(
    model_name: str,
    replacement_lm_name_or_path: str,
    dataloader: DataLoader,
    args: argparse.Namespace,
    text_model: bool = False,
    replace_lm: bool = False,
    return_data: bool = False,
    save_data: bool = True,
    saved_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Extracts head representations from the specified model.

    Args:
        model_name (str): Name of the base model.
        replacement_lm_name_or_path (Optional[str]): Path or name of the replacement language model.
        dataloader (DataLoader): Dataloader containing the dataset.
        args (argparse.Namespace): Parsed command line arguments containing various configurations.

    Returns:
        Dict[str, torch.Tensor]: A dictionary mapping 'layer_{i}/head_{j}' to their corresponding representations.
    """

    # For appropriate filename and save directory
    model_for_representations_extraction = (
        model_name if not replace_lm else replacement_lm_name_or_path
    )

    if (
        not text_model
        and model_name != model_for_representations_extraction
        and replace_lm
    ):
        args.replacement_lm_name_or_path = replacement_lm_name_or_path or model_name

    print(f"\n=== Processing model: {model_for_representations_extraction} ===")

    save_dir = setup_directories(
        representations_dir=args.representations_dir,
        model_name=model_for_representations_extraction,
        dataset_name=args.dataset_name,
        split=args.split,
    )
    # Add chat_mode for create_filename (optional) and the format_prompts functions ("necessary")
    args.chat_mode = True
    save_dir = os.path.join(save_dir, "heads_representations")
    os.makedirs(save_dir, exist_ok=True)
    save_filename = create_filename(args)
    save_path = os.path.join(save_dir, save_filename)
    print(f"Saving representations to: {save_path}")

    # Check if file already exists to avoid duplicate work
    if os.path.exists(save_path):
        print(f"File already exists at {save_path}. Skipping extraction.")
        return save_path

    model, processor = load_hf_model_and_processor_or_tokenizer(
        model_name_or_path=model_name,
        cache_dir=args.model_cache_dir,
        text_model=text_model,
        dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    if args.batch_size > 1:
        if text_model:
            processor.padding_side = "left"
            if processor.pad_token is None:
                processor.add_special_tokens({"pad_token": "[PAD]"})
                model.resize_token_embeddings(len(processor))
        else:
            processor.tokenizer.padding_side = "left"
            if processor.tokenizer.pad_token is None:
                processor.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model.language_model.resize_token_embeddings(len(processor.tokenizer))

    if replacement_lm_name_or_path is not None and replace_lm:
        model = replace_multimodal_lm(
            multimodal_model=model,
            replacement_lm_name_or_path=replacement_lm_name_or_path,
            cache_dir=args.model_cache_dir,
        )

    tracer = HeadProjectionTracer(
        model=model,
        target_layers=args.layer_index,
        target_heads=None,
        tokens_mode=args.tokens_mode,
    )

    tracer.trace()

    # Initialize a dictionary to hold all projections
    all_projections = {}

    num_samples = len(dataloader.dataset)
    progress_bar = tqdm(total=num_samples, desc="Processing samples", unit="sample")

    with torch.no_grad():
        for batch in dataloader:
            questions_and_options = batch["questions"]
            batch_size = len(questions_and_options)
            images = batch.get("images", None)

            full_prompts = format_prompts(
                questions_and_options=questions_and_options,
                images=images,
                args=args,
                processor=processor,
                chat_template_exists=True,
                # !"enforced" but consistent for these experiments
            )

            processor_kwargs = {
                "text": full_prompts,
                "padding": (True if args.batch_size > 1 else False),
                "return_tensors": "pt",
            }

            if text_model or all(image is None for image in images):
                model_inputs = processor(
                    **processor_kwargs,
                ).to(model.device)
            else:
                model_inputs = processor(
                    **processor_kwargs,
                    images=images,
                ).to(model.device)

            _ = model(**model_inputs)

            for key, proj in tracer.get_residual_stream_projections().items():
                all_projections.setdefault(key, []).append(proj.cpu())

            tracer.residual_stream_projections.clear()

            del (
                questions_and_options,
                images,
                full_prompts,
                processor_kwargs,
                model_inputs,
            )
            gc.collect()
            torch.cuda.empty_cache()
            progress_bar.update(batch_size)

    progress_bar.close()

    final_projections_to_save = {}
    # Use list(all_projections.keys()) to create a copy, allowing you to delete from the original dict while iterating
    for key in list(all_projections.keys()):
        # Concatenate the tensors for the current key
        concatenated_tensor = torch.cat(all_projections[key], dim=0)
        final_projections_to_save[key] = concatenated_tensor

        # Delete the list of smaller tensors for that key to free up memory immediately
        del all_projections[key]
        gc.collect()  # Explicitly ask Python's garbage collector to run

    tracer.clear()
    del (
        model,
        processor,
        tracer,
        all_projections,
    )  # all_projections is now empty but we delete it anyway
    gc.collect()
    torch.cuda.empty_cache()

    if save_data:
        save_extracted_residual_stream_data(
            save_path, {"residual_stream_data": final_projections_to_save}
        )
        saved_paths[model_for_representations_extraction] = save_path
        del final_projections_to_save
        gc.collect()

    if return_data:
        return final_projections_to_save


def main():
    """Main function to parse arguments and extract LLMs heads representations for multiple models."""
    parser = argparse.ArgumentParser(
        description="LLMs heads representations extraction in residual stream from models"
    )
    parser.add_argument("--representations-dir", type=str, required=True)
    parser.add_argument("--model-cache-dir", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--dataset-cache-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--layer-index", type=parse_layer_index, default=-1)
    parser.add_argument("--tokens-mode", type=parse_tokens_mode, default="last")
    parser.add_argument("--texts_qa", action="store_true", default=False)
    parser.add_argument("--images_qa", action="store_true", default=False)
    parser.add_argument("--mm-name-or-path", type=str, required=True)
    parser.add_argument("--lm-name-or-path", type=str, required=True)
    parser.add_argument(
        "--question-instruction-type", type=parse_question_instruction, default=None
    )
    parser.add_argument("--continue-final-message", action="store_true", default=False)
    parser.add_argument("--guide-text", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=1)
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
        # {"model_name": args.mm_name_or_path, "replace_lm": False},
        {"model_name": args.lm_name_or_path, "replace_lm": True},
    ]

    # Prepare the dataloader
    print("Preparing dataloader...")
    dataloader = get_dataloader(
        dataset_path_or_name=args.dataset_name,
        cache_dir=args.dataset_cache_dir,
        split=args.split,
        texts_qa=args.texts_qa,
        images_qa=args.images_qa,
        batch_size=args.batch_size,
        question_instruction_type=args.question_instruction_type,
    )

    # Extract and save for each model
    saved_paths = {}
    for param in parameters:
        model_for_representations_extraction = param["model_name"]
        try:
            start_time = time.time()
            heads_representations_extractor(
                model_name=args.mm_name_or_path,
                replacement_lm_name_or_path=args.lm_name_or_path,
                dataloader=dataloader,
                args=args,
                replace_lm=param["replace_lm"],
                save_data=True,
                saved_paths=saved_paths,
            )
            elapsed_time = time.time() - start_time
            print(f"✅ Completed extraction for {model_for_representations_extraction}")
            print(f"⏱️ Time taken: {elapsed_time / 60:.2f} minutes")

        except Exception as e:
            print(
                f"❌ Failed extraction for {model_for_representations_extraction}: {str(e)}"
            )
            traceback.print_exc()

    # Print summary of saved paths
    if all(saved_paths.values()):
        print("\n=== Extraction Complete! ===")
        print("Saved representations:")
        for model_name, path in saved_paths.items():
            print(f"  {model_name}: {path}")


if __name__ == "__main__":
    main()
