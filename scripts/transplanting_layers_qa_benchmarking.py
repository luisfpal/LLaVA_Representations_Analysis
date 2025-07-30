from utils import (
    get_dataloader,
    benchmark_model_vqa_processed_dataloader,
)
from src.transplantation_benchmarking_common import (
    run_transplantation_benchmarking,
    create_common_parser,
    setup_transplantation_methods,
)

COCOQA_DATASET_ARGS = {
    "guide_text": "Answer the question using a single word or phrase.\n",
    "downsample_size": 2500,
    "answer_letters_with_processed_batch": True,
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


def main():
    """
    Benchmark model performance with layer transplantation between multimodal models.

    This script evaluates how transplanting layers from a source multimodal model
    to a target model (with pretrained connector) affects VQA performance.

    Two transplantation methods are supported:
    - sliding_window: Transplant consecutive layers in a sliding window pattern
    - two_parts: Split model at a layer and transplant the first part

    Results are saved as CSV files in the following structure:
    results/transplanting_layers_qa_benchmarking/{dataset_name}/
    ├── {models}_sliding_window_ws{window_size}_s{stride}.csv
    └── {models}_two_parts_s{stride}.csv

    Each CSV contains accuracy metrics for different transplantation configurations
    plus baseline results for both the original multimodal model and the model
    with pretrained connector.
    """

    parser = create_common_parser()
    args = parser.parse_args()
    
    # Setup transplantation methods configuration
    setup_transplantation_methods(args)

    run_transplantation_benchmarking(
        args=args,
        dataset_config=DATASETS,
        get_dataloader_func=get_dataloader,
        benchmark_func=benchmark_model_vqa_processed_dataloader,
        results_subdir="transplanting_layers_qa_benchmarking",
    )


if __name__ == "__main__":
    main()
