import torch
import argparse
import os
import gc
from safetensors.torch import save_file
from typing import Union
from src.residual_stream_tracer import residual_stream_tracer
from utils import (
    setup_multimodal_model,
    get_dataloader,
    seed_all,
)
from utils.metrics_utils import (
    compute_layers_residual_stream_similarities,
    compute_heads_projection_residual_stream_similarities,
    compute_layers_intrinsic_dimension,
    compute_layers_residual_stream_entropy,
)

# Use float16 for relatively light weight memory usage and fast extraction
MODEL_DTYPE = torch.float16
ATTN_IMPLEMENTATION = (
    "flash_attention_2"  # should be changed to "sdpa" for torch.float32
)


RESIDUAL_STREAM_TYPE: set[str] = {
    "output_layer",
    "post_mlp",
    "heads_projection",
}

TOKENS_POOLING_METHOD: set[Union[str, None]] = {
    "mean",
    "last",
    None,
}

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

MODELS = {
    "multimodal_model": {
        "model_name_or_path": "llava-hf/llava-1.5-7b-hf",
    },
    "multimodal_model_pretrained_connector": {
        "model_name_or_path": "llava-hf/llava-1.5-7b-hf",
        "replace_pretrained_projector": True,
        "pretrained_projector_name_or_path": "liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5",
        "replace_language_model": True,
        "language_model_name_or_path": "lmsys/vicuna-7b-v1.5",
    },
}


def main():
    """
    Organize the results in the following format in the results directory of the root of the project:
    - results/
        - dataset_name/
            - similarities/
                - llava-1.5-7b-hf_vs_vicuna-7b-v1.5_<tokens_pooling_method>_<residual_stream_type>.safetensors
            - models_data_measures/
                - model_name/
                    - dataset_entropy_<tokens_pooling_method>_<residual_stream_type>.safetensors
                    - id_<id_nn_rank>_<id_nn_range_max>_<tokens_pooling_method>_<residual_stream_type>.safetensors
                    - prompt_entropy_<tokens_pooling_method>_<residual_stream_type>.safetensors
                    - dataset_entropy_<tokens_pooling_method>_<residual_stream_type>.safetensors
    TODO:
        - I must parse a list of the residual stream types to be extracted from the set RESIDUAL_STREAM_TYPE
        - The same for the tokens pooling methods so that I can select the combinations of residual stream types and tokens pooling methods to be extracted
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-parent-dir", type=str, required=True)
    parser.add_argument("--model-cache-dir", type=str, required=True)
    parser.add_argument("--dataset-name-or-path", type=str, default=None)
    parser.add_argument("--maxk", type=int, default=30)
    parser.add_argument("--accept-rate", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id-nn-rank", type=int, default=16)
    parser.add_argument("--id-nn-range-max", type=int, default=100)
    args = parser.parse_args()

    seed_all(args.seed)
    base_dir = os.path.expanduser(args.result_parent_dir)
    # parse a list of residual stream types
    return_deepcopy = True
    for residual_stream_type in RESIDUAL_STREAM_TYPE:
        for tokens_pooling_method in TOKENS_POOLING_METHOD:
            if tokens_pooling_method == "none":
                return_deepcopy = False
            if (
                residual_stream_type == "heads_projection"
                and tokens_pooling_method == "none"
            ):
                continue
            if residual_stream_type == "post_mlp" and tokens_pooling_method == "none":
                continue
            for dataset_name, dataset_args in DATASETS.items():
                dataloader = get_dataloader(
                    **{
                        **dataset_args,
                        "dataset_name_or_path": args.dataset_name_or_path,
                    }
                )
                residual_stream_multimodal_model = None
                residual_stream_multimodal_model_pretrained_connector = None
                # shapes: (num_layers, num_samples, num_heads, hidden_size)
                # or (num_layers, num_samples, hidden_size)
                common_dir = os.path.join(
                    base_dir,
                    "results",
                    dataset_name,
                )
                similarities_dir = os.path.join(common_dir, "similarities")
                os.makedirs(similarities_dir, exist_ok=True)
                models_data_measures_dir = os.path.join(
                    common_dir, "models_data_measures"
                )
                os.makedirs(models_data_measures_dir, exist_ok=True)

                for model_name, model_args in MODELS.items():
                    mm_model_name = (
                        model_args["model_name_or_path"]
                        .split("/")[-1]
                        .replace("-", "_")
                    )
                    lm_model_name = (
                        model_args.get("language_model_name_or_path", None)
                        .split("/")[-1]
                        .replace("-", "_")
                    )
                    if lm_model_name is None:
                        lm_model_name = mm_model_name
                    model_name = f"{mm_model_name}_vs_{lm_model_name}"

                    model_dir = os.path.join(models_data_measures_dir, model_name)
                    os.makedirs(model_dir, exist_ok=True)

                    model, processor = setup_multimodal_model(
                        **{
                            **model_args,
                            "model_cache_dir": args.model_cache_dir,
                        }
                    )
                    residual_stream = residual_stream_tracer(
                        model=model,
                        processor=processor,
                        dataloader=dataloader,
                        residual_stream_type=residual_stream_type,
                        tokens_pooling_method=tokens_pooling_method,
                        return_deepcopy=return_deepcopy,
                    )
                    if model_name == "multimodal_model":
                        residual_stream_multimodal_model = residual_stream
                    elif model_name == "multimodal_model_pretrained_connector":
                        residual_stream_multimodal_model_pretrained_connector = (
                            residual_stream
                        )

                    if (
                        residual_stream_type == "output_layer"
                        or residual_stream_type == "post_mlp"
                    ):
                        if tokens_pooling_method is not None:
                            layers_intrinsic_dimension = (
                                compute_layers_intrinsic_dimension(
                                    residual_stream,
                                    algorithm="scaling_grid",
                                    k=args.id_nn_rank,
                                    range_max=args.id_nn_range_max,
                                )
                            )
                            save_file(
                                {
                                    "layers_intrinsic_dimension": layers_intrinsic_dimension
                                },
                                os.path.join(
                                    model_dir,
                                    f"id_{args.id_nn_rank}_{args.id_nn_range_max}_{tokens_pooling_method}_{residual_stream_type}.safetensors",
                                ),
                            )
                            layers_dataset_entropy = (
                                compute_layers_residual_stream_entropy(
                                    residual_stream,
                                    entropy_type="dataset-entropy",
                                )
                            )
                            save_file(
                                {"layers_dataset_entropy": layers_dataset_entropy},
                                os.path.join(
                                    model_dir,
                                    f"dataset_entropy_{tokens_pooling_method}_{residual_stream_type}.safetensors",
                                ),
                            )
                        else:
                            layers_prompt_entropy = (
                                compute_layers_residual_stream_entropy(
                                    residual_stream,
                                    entropy_type="prompt-entropy",
                                )
                            )
                            save_file(
                                {"layers_prompt_entropy": layers_prompt_entropy},
                                os.path.join(
                                    model_dir,
                                    f"prompt_entropy_{tokens_pooling_method}_{residual_stream_type}.safetensors",
                                ),
                            )

                    del model, processor
                    gc.collect()
                    torch.cuda.empty_cache()

                if residual_stream_type == "heads_projection":
                    compute_residual_stream_similarities = (
                        compute_heads_projection_residual_stream_similarities
                    )
                else:
                    compute_residual_stream_similarities = (
                        compute_layers_residual_stream_similarities
                    )

                residual_stream_similarities = compute_residual_stream_similarities(
                    residual_stream_multimodal_model,
                    residual_stream_multimodal_model_pretrained_connector,
                    similarity_measures=["neighborhood_overlap", "linear_cka", "svcca"],
                    maxk=args.maxk,
                    accept_rate=args.accept_rate,
                )
                save_file(
                    residual_stream_similarities,
                    os.path.join(
                        similarities_dir,
                        f"llava-1.5-7b-hf_vs_vicuna-7b-v1.5_{tokens_pooling_method}_{residual_stream_type}.safetensors",
                    ),
                )

                del (
                    dataloader,
                    residual_stream_multimodal_model,
                    residual_stream_multimodal_model_pretrained_connector,
                )
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
