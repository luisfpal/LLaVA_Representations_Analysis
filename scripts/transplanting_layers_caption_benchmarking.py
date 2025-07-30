from utils import (
    get_dataloader_for_captioning,
    benchmark_model_captioning_processed_dataloader,
)
from src.transplantation_benchmarking_common import (
    run_transplantation_benchmarking,
    create_common_parser,
    setup_transplantation_methods,
)

# Single dataset configuration for captioning
DATASETS = {
    "coco_captioning": {
        "downsample_size": 2500,
        "return_captions": True,
    }
}


def main():
    """
    Benchmark model performance with layer transplantation between multimodal models.

    This script evaluates how transplanting layers from a source multimodal model
    to a target model (with pretrained connector) affects image captioning performance.

    Two transplantation methods are supported:
    - sliding_window: Transplant consecutive layers in a sliding window pattern
    - two_parts: Split model at a layer and transplant the first part

    Results are saved as CSV files in the following structure:
    results/transplanting_layers_caption_benchmarking/{dataset_name}/
    ├── {models}_sliding_window_ws{window_size}_s{stride}.csv
    └── {models}_two_parts_s{stride}.csv

    Each CSV contains captioning metrics (BLEU, METEOR, ROUGE-L, CIDEr, SPICE)
    for different transplantation configurations plus baseline results for both
    the original multimodal model and the model with pretrained connector.
    """

    parser = create_common_parser()
    # Override default max_new_tokens for captioning
    parser.set_defaults(max_new_tokens=50)
    args = parser.parse_args()

    # Setup transplantation methods configuration
    setup_transplantation_methods(args)

    run_transplantation_benchmarking(
        args=args,
        dataset_config=DATASETS,
        get_dataloader_func=get_dataloader_for_captioning,
        benchmark_func=benchmark_model_captioning_processed_dataloader,
        results_subdir="transplanting_layers_caption_benchmarking",
    )


if __name__ == "__main__":
    main()
