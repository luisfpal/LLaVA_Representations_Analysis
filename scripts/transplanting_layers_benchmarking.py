import os
import argparse
from typing import Dict
import time
import pandas as pd
from utils import (
    get_dataloader,
    setup_multimodal_model,
    benchmark_model_vqa_processed_dataloader,
    transplant_layers_weights,
)
from utils.operations_utils import seed_all

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
}
TRANSPLANTATION_WEIGHTS_METHODS = {
    "sliding_window": {
        "stride": 2,
        "window_size": 2,
    },
    "two_parts": {
        "stride": 2,
    },
}


def main():
    """
    Benchmark model performance with layer transplantation between multimodal models.

    This script evaluates how transplanting layers from a source multimodal model
    to a target model (with pretrained connector) affects VQA performance.

    Two transplantation methods are supported:
    - sliding_window: Transplant consecutive layers in a sliding window pattern
    - two_parts: Split model at a layer and transplant the first part

    Results are saved as CSV files in the following structure:
    results/transplanting_layers_benchmarking/{dataset_name}/
    ├── {models}_sliding_window_ws{window_size}_s{stride}.csv
    └── {models}_two_parts_s{stride}.csv

    Each CSV contains accuracy metrics for different transplantation configurations
    plus baseline results for both the original multimodal model and the model
    with pretrained connector.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal_model_name_or_path", type=str, required=True)
    parser.add_argument("--model_cache_dir", type=str, required=True)
    parser.add_argument("--pretrained_projector_name_or_path", type=str, required=True)
    parser.add_argument("--language_model_name_or_path", type=str, required=True)
    parser.add_argument(
        "--transplantation_method",
        type=str,
        required=True,
        choices=list(TRANSPLANTATION_WEIGHTS_METHODS.keys()),
    )
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--max_new_tokens", type=int, default=1)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--dataset_name_or_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_all(args.seed)

    # Setup models
    multimodal_model, processor = setup_multimodal_model(
        multimodal_model_name_or_path=args.multimodal_model_name_or_path,
        model_cache_dir=args.model_cache_dir,
        device_map="cpu",
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
    )

    model_num_layers = multimodal_model.language_model.config.num_hidden_layers
    models_name = (
        f"{args.multimodal_model_name_or_path.split('/')[-1]}"
        f"_{args.language_model_name_or_path.split('/')[-1]}"
    )

    # Setup output directories
    results_dir = os.path.expanduser(args.results_dir)
    benchmarking_results_dir = os.path.join(
        results_dir, "transplanting_layers_benchmarking"
    )
    os.makedirs(benchmarking_results_dir, exist_ok=True)

    # Process each dataset
    for dataset_name, dataset_args in DATASETS.items():
        print(f"\n{'=' * 100}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'=' * 100}\n")

        processed_dataloader = get_dataloader(
            **{
                **dataset_args,
                "dataset_name_or_path": args.dataset_name_or_path,
                "processor": processor,
                "answer_letters_with_processed_batch": True,
                "batch_size": args.batch_size,
                "seed": args.seed,
            }
        )

        dataset_dir = os.path.join(benchmarking_results_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        # Benchmark baseline models once per dataset
        print("Benchmarking baseline multimodal model...")
        multimodal_model.to("cuda:0")
        mm_model_results = benchmark_model_vqa_processed_dataloader(
            model=multimodal_model,
            processed_dataloader=processed_dataloader,
            processor=processor,
            max_new_tokens=args.max_new_tokens,
        )
        multimodal_model.to("cpu")

        print("Benchmarking multimodal model with pretrained connector...")
        multimodal_model_pretrained_connector.to("cuda:0")
        mm_pretrained_connector_results = benchmark_model_vqa_processed_dataloader(
            model=multimodal_model_pretrained_connector,
            processed_dataloader=processed_dataloader,
            processor=processor,
            max_new_tokens=args.max_new_tokens,
        )
        multimodal_model_pretrained_connector.to("cpu")

        # Process selected transplantation method
        transplantation_method_args = TRANSPLANTATION_WEIGHTS_METHODS[
            args.transplantation_method
        ]
        benchmarking_results: Dict[str, Dict[str, float]] = {}
        stride = transplantation_method_args["stride"]
        layers = [i * stride for i in range(model_num_layers // stride)]

        start_time = time.time()

        if args.transplantation_method == "sliding_window":
            window_size = transplantation_method_args["window_size"]
            filename = f"{models_name}_sliding_window_ws{window_size}_s{stride}.csv"

            print(
                f"\nRunning sliding window transplantation (window_size={window_size}, stride={stride})..."
            )
            for start_layer in layers:
                print(f"  Testing start_layer={start_layer}")
                model_with_transplanted_layers = transplant_layers_weights(
                    source_model=multimodal_model,
                    target_model=multimodal_model_pretrained_connector,
                    method=args.transplantation_method,
                    start_layer=start_layer,
                    window_size=window_size,
                    in_place_transplantation=False,
                )

                model_with_transplanted_layers.to("cuda:0")
                results = benchmark_model_vqa_processed_dataloader(
                    model=model_with_transplanted_layers,
                    processed_dataloader=processed_dataloader,
                    processor=processor,
                    max_new_tokens=args.max_new_tokens,
                )
                benchmarking_results[f"start_layer_{start_layer}"] = results
                model_with_transplanted_layers.to("cpu")
                del model_with_transplanted_layers

        elif args.transplantation_method == "two_parts":
            filename = f"{models_name}_two_parts_s{stride}.csv"
            layers = [layer_idx - 1 for layer_idx in layers][:-1]  # Exclude last layer

            print(f"\nRunning two parts transplantation (stride={stride})...")
            for split_layer in layers:
                print(f"  Testing split_layer={split_layer}")
                model_with_transplanted_layers = transplant_layers_weights(
                    source_model=multimodal_model,
                    target_model=multimodal_model_pretrained_connector,
                    method=args.transplantation_method,
                    split_layer=split_layer,
                    in_place_transplantation=False,
                )

                model_with_transplanted_layers.to("cuda:0")
                results = benchmark_model_vqa_processed_dataloader(
                    model=model_with_transplanted_layers,
                    processed_dataloader=processed_dataloader,
                    processor=processor,
                    max_new_tokens=args.max_new_tokens,
                )
                benchmarking_results[f"split_layer_{split_layer}"] = results
                model_with_transplanted_layers.to("cpu")
                del model_with_transplanted_layers

        end_time = time.time()
        print(
            f"Time taken for {args.transplantation_method} transplantation: {(end_time - start_time) / 60:.2f} minutes"
        )

        # Add baseline results and save
        benchmarking_results["mm_model"] = mm_model_results
        benchmarking_results["mm_pretrained_connector"] = (
            mm_pretrained_connector_results
        )

        df = pd.DataFrame.from_dict(benchmarking_results, orient="index")
        output_path = os.path.join(dataset_dir, filename)
        df.to_csv(output_path, index=True)
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
