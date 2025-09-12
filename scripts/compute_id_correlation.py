import torch
import argparse
import os
import gc
from safetensors.torch import save_file
from src.residual_stream_tracer import residual_stream_tracer
from utils import (
    setup_multimodal_model,
    get_dataloader,
    get_dataloader_for_captioning,
    seed_all,
)
from utils.id_correlation import compute_layers_id_correlation

# Use float16 for relatively light weight memory usage and fast extraction
MODEL_DTYPE = torch.float16

# Dataset configurations
COCOQA_DATASET_ARGS = {
    "guide_text": "Answer the question using a single word or phrase.\n",
    "downsample_size": 2500,
}

DATASETS = {
    "cocoqa_txt": {
        "texts_qa": True,
        **COCOQA_DATASET_ARGS,
    },
    "cocoqa_img": {
        "images_qa": True,
        **COCOQA_DATASET_ARGS,
    },
    "coco_captioning": {
        "downsample_size": 2500,
    },
}

MODELS = {
    "multimodal_model": {
        "multimodal_model_name_or_path": "llava-hf/llava-1.5-7b-hf",
    },
    "multimodal_model_pretrained_connector": {
        "multimodal_model_name_or_path": "llava-hf/llava-1.5-7b-hf",
        "replace_pretrained_projector": True,
        "pretrained_projector_name_or_path": "liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5",
        "replace_language_model": True,
        "language_model_name_or_path": "lmsys/vicuna-7b-v1.5",
    },
}


def get_model_identifier(model_args: dict) -> str:
    """
    Generate a standardized model identifier from model args.

    Args:
        model_args: Dictionary containing model configuration

    Returns:
        String identifier for the model
    """
    language_model_path = model_args.get("language_model_name_or_path")
    pretrained_projector_path = model_args.get("pretrained_projector_name_or_path")
    if language_model_path and pretrained_projector_path:
        return pretrained_projector_path.split("/")[-1]
    else:
        return model_args["multimodal_model_name_or_path"].split("/")[-1]


def get_dataloader_for_dataset(dataset_name: str, dataset_args: dict, args) -> object:
    """
    Get the appropriate dataloader function for the given dataset.

    Args:
        dataset_name: Name of the dataset
        dataset_args: Dataset configuration arguments
        args: Command line arguments

    Returns:
        Processed dataloader
    """
    if dataset_name == "coco_captioning":
        get_dataloader_func = get_dataloader_for_captioning
    else:
        get_dataloader_func = get_dataloader

    return get_dataloader_func(
        **{
            **dataset_args,
            "dataset_path_or_name": args.dataset_path_or_name,
            "seed": args.seed,
            "processor": args.processor,
            "batch_size": args.batch_size,
        }
    )


def process_models_for_id_correlation(
    dataset_name: str,
    dataset_args: dict,
    args,
    similarities_dir: str,
) -> None:
    """
    Process both models to compute ID correlation between them.

    Args:
        dataset_name: Name of the dataset
        dataset_args: Dataset configuration arguments
        args: Command line arguments
        similarities_dir: Directory to save similarity results
    """
    print(f"\n{'+' * 60}")
    print("🤖 Processing models for ID correlation")
    print(f"{'+' * 60}")

    # Storage for residual streams from both models
    residual_stream_multimodal_model = None
    residual_stream_multimodal_model_pretrained_connector = None

    # Process each model configuration
    for model_key, model_args in MODELS.items():
        print(f"\n{'-' * 40}")
        print(f"🤖 Processing model: {model_key}")
        print(f"{'-' * 40}")

        # Load model and processor
        model, processor = setup_multimodal_model(
            **{
                **model_args,
                "model_cache_dir": args.model_cache_dir,
            }
        )

        # Get dataloader for the dataset
        args.processor = processor  # Add processor to args for the helper function
        processed_dataloader = get_dataloader_for_dataset(
            dataset_name, dataset_args, args
        )

        # Extract residual stream
        residual_stream = residual_stream_tracer(
            model=model,
            processed_dataloader=processed_dataloader,
            residual_stream_type="output_layer",
            tokens_pooling_method="last",
            return_deepcopy=True,
        )

        # Store residual streams for correlation computation
        if model_key == "multimodal_model":
            residual_stream_multimodal_model = residual_stream
        elif model_key == "multimodal_model_pretrained_connector":
            residual_stream_multimodal_model_pretrained_connector = residual_stream

        # Clean up model from memory
        del model, processor, processed_dataloader
        gc.collect()
        torch.cuda.empty_cache()

    # Compute ID correlation between the two models
    print("🧮 Computing ID correlation between models...")
    layers_id_correlation = compute_layers_id_correlation(
        residual_stream_multimodal_model,
        residual_stream_multimodal_model_pretrained_connector,
        N=args.id_correlation_permutations,
        algorithm="twoNN",
        k=args.id_nn_rank,
        return_pvalue=False,
    )

    # Save ID correlation results
    save_file(
        {"layers_id_correlation": layers_id_correlation},
        os.path.join(
            similarities_dir,
            "llava-1.5-7b-hf_vs_vicuna-7b-v1.5_id_correlation_output_layer_last.safetensors",
        ),
    )

    # Clean up residual streams from memory
    del (
        residual_stream_multimodal_model,
        residual_stream_multimodal_model_pretrained_connector,
    )
    gc.collect()
    torch.cuda.empty_cache()


def main():
    """
    Compute ID correlation between residual streams from multimodal models.

    This script extracts residual streams at the output_layer with last token pooling
    and computes ID correlation measures between the two model configurations.
    """
    parser = argparse.ArgumentParser(
        description="Compute ID correlation between residual streams from multimodal models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--dataset-path-or-name", type=str, required=True)
    parser.add_argument("--model-cache-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)

    # Analysis configuration
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="cocoqa_txt,cocoqa_img",
        help=f"Comma-separated list of dataset types to process. "
        f"Valid options: {', '.join(sorted(DATASETS.keys()))}. "
        f"If not specified, cocoqa_txt and cocoqa_img will be processed.",
    )

    # ID correlation parameters
    parser.add_argument("--id-nn-rank", type=int, default=16)
    parser.add_argument("--id-correlation-permutations", type=int, default=100)

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Validate argument ranges
    if args.id_nn_rank <= 0:
        parser.error("id-nn-rank must be positive")
    if args.id_correlation_permutations <= 0:
        parser.error("id-correlation-permutations must be positive")

    # Set random seed for reproducibility
    seed_all(args.seed)
    results_dir = os.path.expanduser(args.results_dir)

    # Parse dataset types
    dataset_types = [dt.strip() for dt in args.dataset_type.split(",")]
    for dt in dataset_types:
        if dt not in DATASETS:
            parser.error(
                f"Invalid dataset type: {dt}. Valid options: {list(DATASETS.keys())}"
            )

    print(f"🔧 Processing {len(dataset_types)} dataset types: {dataset_types}")

    print(f"🔧 Using ID correlation permutations: {args.id_correlation_permutations}")

    # Process each dataset
    for dataset_name in dataset_types:
        dataset_args = DATASETS[dataset_name]
        print(f"\n{'*' * 60}")
        print(f"📊 Processing dataset: {dataset_name}")
        print(f"{'*' * 60}")

        # Set up output directories
        common_dir = os.path.join(results_dir, dataset_name)
        similarities_dir = os.path.join(common_dir, "similarities")
        os.makedirs(similarities_dir, exist_ok=True)

        # Process models for ID correlation computation
        process_models_for_id_correlation(
            dataset_name, dataset_args, args, similarities_dir
        )

    print(f"\n{'=' * 80}")
    print("🎉 ID correlation computation completed successfully!")
    print(f"📁 Results saved to: {results_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
