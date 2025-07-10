import os
import argparse
import pandas as pd
from utils import (
    get_dataloader,
    setup_multimodal_model,
    benchmark_model_vqa_processed_dataloader,
    transplant_layers_weights,
)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal_model_name_or_path", type=str, required=True)
    parser.add_argument("--model_cache_dir", type=str, required=True)
    parser.add_argument("--pretrained_projector_name_or_path", type=str, required=True)
    parser.add_argument("--language_model_name_or_path", type=str, required=True)
    parser.add_argument("--dataset-name-or-path", type=str, required=True)
    parser.add_argument("--transplantation_method", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=25)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()

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
    models = (
        f"{args.multimodal_model_name_or_path.split('/')[-1].replace('-', '_')}"
        f"-{args.language_model_name_or_path.split('/')[-1].replace('-', '_')}"
    )

    results_dir = os.path.expanduser(args.results_dir)
    benchmarking_results_dir = os.path.join(
        results_dir, "transplanting_layers_benchmarking"
    )
    os.makedirs(benchmarking_results_dir, exist_ok=True)

    for dataset_name, dataset_args in DATASETS.items():
        processed_dataloader = get_dataloader(
            **{
                **dataset_args,
                "dataset_name_or_path": args.dataset_name_or_path,
                "processor": processor,
                "answer_letters_with_processed_batch": True,
                "batch_size": args.batch_size,
            }
        )
        dataset_dir = os.path.join(benchmarking_results_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        for (
            transplantation_method,
            transplantation_method_args,
        ) in TRANSPLANTATION_WEIGHTS_METHODS.items():
            benchmarking_results = {}
            stride = transplantation_method_args["stride"]
            layers = [i * stride for i in range(model_num_layers // stride)]
            if transplantation_method == "sliding_window":
                window_size = transplantation_method_args["window_size"]
                for start_layer in layers:
                    model_with_transplanted_layers = transplant_layers_weights(
                        source_model=multimodal_model,
                        target_model=multimodal_model_pretrained_connector,
                        method=transplantation_method,
                        start_layer=start_layer,
                        window_size=window_size,
                        in_place_transplantation=False,
                    )

                    model_with_transplanted_layers.to("cuda:0")

                    results = benchmark_model_vqa_processed_dataloader(
                        model=model_with_transplanted_layers,
                        dataloader=processed_dataloader,
                        processor=processor,
                        max_new_tokens=args.max_new_tokens,
                    )
                    benchmarking_results[start_layer] = results
                    # {
                    #     "start_layer": {
                    #         "num_correct_answers": correct_answers,
                    #         "num_incorrect_answers": incorrect_answers,
                    #         "accuracy": correct_answers / (correct_answers + incorrect_answers),
                    #     },
                    # }
                    filename = f"{models}_{transplantation_method}_ws{window_size}_s{stride}.csv"

            elif transplantation_method == "two_parts":
                layers = [layer_idx - 1 for layer_idx in layers][:-1]
                for split_layer in layers:
                    model_with_transplanted_layers = transplant_layers_weights(
                        source_model=multimodal_model,
                        target_model=multimodal_model_pretrained_connector,
                        method=transplantation_method,
                        split_layer=split_layer,
                        in_place_transplantation=False,
                    )

                    model_with_transplanted_layers.to("cuda:0")

                    results = benchmark_model_vqa_processed_dataloader(
                        model=model_with_transplanted_layers,
                        dataloader=processed_dataloader,
                        processor=processor,
                        max_new_tokens=args.max_new_tokens,
                    )
                    benchmarking_results[split_layer] = results
                    # {
                    #     "split_layer": {
                    #         "num_correct_answers": correct_answers,
                    #         "num_incorrect_answers": incorrect_answers,
                    #         "accuracy": correct_answers / (correct_answers + incorrect_answers),
                    #     },
                    # }
                    filename = f"{models}_{transplantation_method}_s{stride}.csv"
            df = pd.DataFrame.from_dict(benchmarking_results, orient="index")
            df.to_csv(
                os.path.join(
                    dataset_dir,
                    filename,
                ),
                index=True,
            )
