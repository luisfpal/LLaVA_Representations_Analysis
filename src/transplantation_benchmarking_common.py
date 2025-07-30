import gc
import os
import argparse
from typing import Dict, List, Tuple, Callable, Any
import time
import pandas as pd
import torch
from utils import setup_multimodal_model, transplant_layers_weights
from utils.operations_utils import seed_all

# Transplantation methods configuration - will be set dynamically based on args
TRANSPLANTATION_WEIGHTS_METHODS = {}


# Remove the hardcoded benchmark_model function - it should be passed as a parameter


def get_transplantation_layers(method: str, num_layers: int, stride: int) -> List[int]:
    """Get the list of layers to process for transplantation."""
    base_layers = [i * stride for i in range(num_layers // stride + 1)]
    if method == "two_parts":
        return [layer_idx - 1 for layer_idx in base_layers][1:]
    return base_layers[:-1]


def get_transplantation_params(
    method: str, layer_value: int, method_args: Dict
) -> Dict:
    """Get the parameters for transplant_layers_weights based on method."""
    if method == "sliding_window":
        return {
            "start_layer": layer_value,
            "window_size": method_args["window_size"],
        }
    else:  # two_parts
        return {"split_layer": layer_value}


def run_transplantation_experiments(
    method: str,
    method_args: Dict,
    source_model,
    target_model,
    num_layers: int,
    processed_dataloader,
    processor,
    max_new_tokens: int,
    benchmark_func: Callable,
) -> Dict[str, Dict[str, float]]:
    """Run transplantation experiments for the specified method."""
    stride = method_args["stride"]
    layers = get_transplantation_layers(method, num_layers, stride)
    benchmarking_results = {}

    print(f"\n{'+-' * 40}")
    print(f"Running {method} transplantation... 🔄")
    print(f"Number of layers transplantations: {len(layers)}")
    print(f"{'+-' * 40}\n")
    
    for idx, layer_value in enumerate(layers):
        progress = (idx + 1) / len(layers)
        print(
            f"\nProcessing layer {layer_value}: [{idx + 1}/{len(layers)}] ({progress:.2%})"
        )

        transplant_params = get_transplantation_params(method, layer_value, method_args)

        model_with_transplanted_layers = transplant_layers_weights(
            source_model=source_model,
            target_model=target_model,
            method=method,
            **transplant_params,
            in_place_transplantation=False,
        )

        results = benchmark_func(
            model_with_transplanted_layers,
            processed_dataloader,
            processor,
            max_new_tokens,
        )

        key = (
            f"{'start' if method == 'sliding_window' else 'split'}_layer_{layer_value}"
        )
        benchmarking_results[key] = results

        del model_with_transplanted_layers
        gc.collect()
        torch.cuda.empty_cache()

    return benchmarking_results


def get_output_filename(models_name: str, method: str, method_args: Dict) -> str:
    """Generate output filename based on transplantation method."""
    stride = method_args["stride"]
    if method == "sliding_window":
        window_size = method_args["window_size"]
        return f"{models_name}_sliding_window_ws{window_size}_s{stride}.csv"
    else:  # two_parts
        return f"{models_name}_two_parts_s{stride}.csv"


def setup_models(args) -> Tuple[object, object, object]:
    """Setup and return the multimodal models and processor."""
    multimodal_model, processor = setup_multimodal_model(
        multimodal_model_name_or_path=args.multimodal_model_name_or_path,
        model_cache_dir=args.model_cache_dir,
        device_map="cpu",
        attn_implementation="sdpa",
        model_dtype=torch.float32,
    )

    multimodal_model_pretrained_connector = setup_multimodal_model(
        multimodal_model_name_or_path=args.multimodal_model_name_or_path,
        model_cache_dir=args.model_cache_dir,
        replace_pretrained_projector=True,
        pretrained_projector_name_or_path=args.pretrained_projector_name_or_path,
        replace_language_model=True,
        language_model_name_or_path=args.language_model_name_or_path,
        skip_processor=True,
        device_map="cpu",
        attn_implementation="sdpa",
        model_dtype=torch.float32,
    )

    return multimodal_model, multimodal_model_pretrained_connector, processor


def benchmark_baseline_models(
    multimodal_model,
    multimodal_model_pretrained_connector,
    processed_dataloader,
    processor,
    max_new_tokens: int,
    benchmark_func: Callable,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Benchmark both baseline models and return their results."""
    print("\nBenchmarking baseline multimodal model... 📊")
    mm_model_results = benchmark_func(
        multimodal_model,
        processed_dataloader,
        processor,
        max_new_tokens,
        dtype=torch.float32,
    )

    print("\nBenchmarking multimodal model with pretrained connector... 📊")
    mm_pretrained_connector_results = benchmark_func(
        multimodal_model_pretrained_connector,
        processed_dataloader,
        processor,
        max_new_tokens,
        dtype=torch.float32,
    )

    return mm_model_results, mm_pretrained_connector_results


def create_benchmark_model_wrapper(benchmark_func: Callable) -> Callable:
    """Create a wrapper for the benchmark function with GPU device management."""
    def benchmark_model(
        model,
        processed_dataloader,
        processor,
        max_new_tokens: int,
        dtype: torch.dtype = torch.float16,
    ) -> Dict[str, float]:
        """Benchmark model with automatic GPU device management."""
        model.to(device="cuda:0", dtype=dtype)
        results = benchmark_func(
            model=model,
            processed_dataloader=processed_dataloader,
            processor=processor,
            max_new_tokens=max_new_tokens,
        )
        model.to("cpu")
        return results
    
    return benchmark_model


def run_transplantation_benchmarking(
    args: argparse.Namespace,
    dataset_config: Dict[str, Any],
    get_dataloader_func: Callable,
    benchmark_func: Callable,
    results_subdir: str,
) -> None:
    """
    Run transplantation benchmarking experiments.
    
    Args:
        args: Command line arguments
        dataset_config: Configuration for the dataset
        get_dataloader_func: Function to get the dataloader
        benchmark_func: Function to benchmark the model
        results_subdir: Subdirectory name for results
    """
    seed_all(args.seed)

    # Setup models
    multimodal_model, multimodal_model_pretrained_connector, processor = setup_models(args)
    model_num_layers = multimodal_model.language_model.config.num_hidden_layers
    models_name = (
        f"{args.multimodal_model_name_or_path.split('/')[-1]}"
        f"_{args.language_model_name_or_path.split('/')[-1]}"
    )

    # Setup output directories
    results_dir = os.path.expanduser(args.results_dir)
    benchmarking_results_dir = os.path.join(results_dir, results_subdir)
    os.makedirs(benchmarking_results_dir, exist_ok=True)

    # Create benchmark model wrapper
    benchmark_model = create_benchmark_model_wrapper(benchmark_func)

    # Process each dataset
    for dataset_name, dataset_args in dataset_config.items():
        print(f"\n{'=' * 80}")
        print(f"Processing dataset: {dataset_name} 📊")
        print(f"{'=' * 80}\n")

        # Prepare dataloader arguments
        dataloader_kwargs = {
            **dataset_args,
            "dataset_path_or_name": args.dataset_path_or_name,
            "processor": processor,
            "batch_size": args.batch_size,
            "seed": args.seed,
        }
        
        processed_dataloader = get_dataloader_func(**dataloader_kwargs)

        dataset_dir = os.path.join(benchmarking_results_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        # Benchmark baseline models
        mm_model_results, mm_pretrained_connector_results = benchmark_baseline_models(
            multimodal_model,
            multimodal_model_pretrained_connector,
            processed_dataloader,
            processor,
            args.max_new_tokens,
            benchmark_model,
        )

        # Run transplantation experiments
        transplantation_method_args = TRANSPLANTATION_WEIGHTS_METHODS[
            args.transplantation_method
        ]

        start_time = time.time()
        benchmarking_results = run_transplantation_experiments(
            method=args.transplantation_method,
            method_args=transplantation_method_args,
            source_model=multimodal_model,
            target_model=multimodal_model_pretrained_connector,
            num_layers=model_num_layers,
            processed_dataloader=processed_dataloader,
            processor=processor,
            max_new_tokens=args.max_new_tokens,
            benchmark_func=benchmark_model,
        )
        end_time = time.time()

        print(
            f"Time taken for {args.transplantation_method} transplantation: {(end_time - start_time) / 60:.2f} minutes 🕒"
        )

        # Add baseline results and save
        benchmarking_results.update(
            {
                "mm_model": mm_model_results,
                "mm_pretrained_connector": mm_pretrained_connector_results,
            }
        )

        filename = get_output_filename(
            models_name, args.transplantation_method, transplantation_method_args
        )
        df = pd.DataFrame.from_dict(benchmarking_results, orient="index")
        output_path = os.path.join(dataset_dir, filename)
        df.to_csv(output_path, index=True)
        print(f"Results saved to: {output_path} 📝")
        print(f"{'=' * 80}\n")


def create_common_parser() -> argparse.ArgumentParser:
    """Create a common argument parser for transplantation benchmarking."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal_model_name_or_path", type=str, required=True)
    parser.add_argument("--model_cache_dir", type=str, required=True)
    parser.add_argument("--pretrained_projector_name_or_path", type=str, required=True)
    parser.add_argument("--language_model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--transplantation_method",
        type=str,
        required=True,
        choices=["sliding_window", "two_parts"],
    )
    parser.add_argument("--stride", type=int, default=2, 
                       help="Stride value for transplantation methods")
    parser.add_argument("--window_size", type=int, default=2,
                       help="Window size for sliding_window method (ignored for two_parts)")
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--dataset_path_or_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def setup_transplantation_methods(args: argparse.Namespace) -> None:
    """Setup transplantation methods configuration based on command line arguments."""
    global TRANSPLANTATION_WEIGHTS_METHODS
    
    TRANSPLANTATION_WEIGHTS_METHODS = {
        "sliding_window": {
            "stride": args.stride,
            "window_size": args.window_size,
        },
        "two_parts": {
            "stride": args.stride,
        },
    } 