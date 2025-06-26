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
    replace_multimodal_projector,
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
    pretrained_projector_name_or_path: Optional[str] = None,
    replace_projector: bool = False,
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
        text_model (bool): Whether the model is a text model.
        replace_lm (bool): Whether to replace the language model.
        pretrained_projector_name_or_path (Optional[str]): Path or name of the pretrained projector.
        replace_projector (bool): Whether to replace the projector.
        return_data (bool): Whether to return the data.
        save_data (bool): Whether to save the data.
        saved_paths (Optional[Dict[str, str]]): Dictionary of saved paths.

    Returns:
        Dict[str, torch.Tensor]: A dictionary mapping 'layer_{i}/head_{j}' to their corresponding representations.
    """

    # For appropriate filename and save directory
    model_for_representations_extraction = (
        model_name if not replace_lm else replacement_lm_name_or_path
    )

    print(f"\n=== Processing model: {model_for_representations_extraction} ===")

    save_dir = setup_directories(
        representations_dir=args.representations_dir,
        model_name=model_for_representations_extraction,
        dataset_name=args.dataset_name,
        split=args.split,
    )
    # Add chat_mode for create_filename (optional) and the format_prompts functions ("necessary")
    save_dir = os.path.join(save_dir, "heads_representations")
    os.makedirs(save_dir, exist_ok=True)
    args.replace_projector = replace_projector  # name consistency
    save_filename = create_filename(args)
    save_path = os.path.join(save_dir, save_filename)

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

    if replace_projector and pretrained_projector_name_or_path is not None:
        model = replace_multimodal_projector(
            multimodal_model=model,
            pretrained_projector_model_name_or_path=pretrained_projector_name_or_path,
            cache_dir=args.model_cache_dir,
        )

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

    # Pre-allocate final tensors instead of using lists
    num_samples = len(dataloader.dataset)
    hidden_size = model.language_model.model.config.hidden_size

    # Initialize pre-allocated tensors for efficient memory usage
    final_projections_to_save = {}
    sample_idx = 0

    # Get tensor keys from first batch to pre-allocate
    first_batch_processed = False

    update_every = max(1, int(0.05 * num_samples))
    progress_bar = tqdm(total=num_samples, desc="Processing samples", unit="sample")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            questions_and_options = batch["questions"]
            images = batch.get("images", None)
            batch_size = len(questions_and_options)

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

            # Direct tensor copying instead of list accumulation
            projections = tracer.get_residual_stream_projections()

            # Pre-allocate tensors on first batch
            if not first_batch_processed:
                print(
                    f"Pre-allocating tensors for {len(projections)} layer-head combinations..."
                )
                for key, proj in projections.items():
                    # Pre-allocate tensor with full dataset size
                    final_projections_to_save[key] = torch.empty(
                        (num_samples, hidden_size), dtype=proj.dtype, device="cpu"
                    )
                first_batch_processed = True
                print(
                    f"Pre-allocated {len(final_projections_to_save)} tensors of shape ({num_samples}, {hidden_size})"
                )

            # Direct indexing instead of concatenation
            for key, proj in projections.items():
                end_idx = sample_idx + batch_size
                final_projections_to_save[key][sample_idx:end_idx] = proj.cpu()

            tracer.residual_stream_projections.clear()
            sample_idx += batch_size

            # Clean up batch-specific variables
            del (
                questions_and_options,
                images,
                full_prompts,
                processor_kwargs,
                model_inputs,
                projections,
            )

            # Less frequent memory cleanup
            if batch_idx % update_every == 0:
                progress_bar.update(update_every)
                if (
                    batch_idx % (update_every * 4) == 0
                ):  # Every 20% instead of every batch
                    gc.collect()
                    torch.cuda.empty_cache()

    progress_bar.close()
    print("Tensor extraction completed. Cleaning up model resources...")

    # Clean up model and tracer
    tracer.clear()
    del model, processor, tracer
    gc.collect()
    torch.cuda.empty_cache()

    if save_data:
        print(f"Saving {len(final_projections_to_save)} tensors to {save_path}...")

        # Efficient saving with memory management
        save_extracted_residual_stream_data(
            save_path, {"residual_stream_data": final_projections_to_save}
        )

        saved_paths[model_for_representations_extraction] = save_path

        if not return_data:
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
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--layer-index", type=parse_layer_index, default=-1)
    parser.add_argument("--tokens-mode", type=parse_tokens_mode, default="last")
    parser.add_argument("--texts_qa", action="store_true", default=False)
    parser.add_argument("--images_qa", action="store_true", default=False)
    parser.add_argument("--mm-name-or-path", type=str, required=True)
    parser.add_argument("--lm-name-or-path", type=str, required=True)
    parser.add_argument("--pretrained-projector-name-or-path", type=str, default=None)
    parser.add_argument(
        "--question-instruction-type", type=parse_question_instruction, default=None
    )
    parser.add_argument("--chat-mode", action="store_true", default=False)
    parser.add_argument("--continue-final-message", action="store_true", default=False)
    parser.add_argument("--guide-text", type=str, default="")
    parser.add_argument("--downsample-size", type=int, default=None)
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
        {
            "model_name": args.mm_name_or_path,
            "replace_lm": False,
            "replace_projector": False,
        },
    ]

    # Add projector replacement variant if specified
    if args.pretrained_projector_name_or_path:
        parameters.append(
            {
                "model_name": args.lm_name_or_path,
                "replace_lm": True,
                "replace_projector": True,
            }
        )

    # Prepare the dataloader
    print("Preparing dataloader...")
    dataloader = get_dataloader(
        dataset_path_or_name=args.dataset_name,
        cache_dir=args.dataset_cache_dir,
        split=args.split,
        texts_qa=args.texts_qa,
        images_qa=args.images_qa,
        question_instruction_type=args.question_instruction_type,
        batch_size=args.batch_size,
        downsample_size=args.downsample_size,
        seed=args.seed,
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
                pretrained_projector_name_or_path=args.pretrained_projector_name_or_path,
                replace_projector=param["replace_projector"],
                save_data=True,
                saved_paths=saved_paths,
            )
            elapsed_time = time.time() - start_time
            print(f"✅ Completed extraction for {model_for_representations_extraction}")
            print(f"⏱️ Time taken: {elapsed_time / 60:.2f} minutes")

            gc.collect()

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
