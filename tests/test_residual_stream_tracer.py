"""
Tests for the residual stream tracer using the specific settings for
the experiments made for this project.
"""

from typing import Dict, Union
import gc
import torch
from src.residual_stream_tracer import residual_stream_tracer
from utils import (
    setup_multimodal_extraction,
    compute_layers_residual_stream_entropy,
    seed_all,
)


def print_heads_representations_info(
    representations: Union[torch.Tensor, Dict[str, torch.Tensor]],
):
    # check the dimensions of the representations in the case of a single tensor, or in the case of a dictionary
    if isinstance(representations, torch.Tensor):
        print(f"Shape of the representations: {representations.shape}")
    elif isinstance(representations, dict):
        print(f"Number of samples: {len(representations)}")
        num_samples = 2
        for idx, (key, value) in enumerate(representations.items()):
            if idx > num_samples:
                break
            print(f"Shape of the representations for sample {idx}: {value.shape}")


multimodal_extraction_args = {
    "multimodal_model_name_or_path": "llava-hf/llava-1.5-7b-hf",
    "model_cache_dir": "~/scratch/huggingface/hub",
    "dataset_path_or_name": "~/scratch/datasets/cocoqa_unified",
    "batch_size": 25,
    "texts_qa": False,
    "images_qa": True,
    "downsample_size": 2500,
    "guide_text": "Answer the question using a single word or phrase.\n",
    "model_dtype": torch.float16,
    "attn_implementation": "flash_attention_2",
    "seed": 42,
}


def test_residual_stream_template(
    residual_stream_type: str, tokens_pooling_method: str, return_deepcopy: bool = True
):
    model, processor, dataloader = setup_multimodal_extraction(
        **multimodal_extraction_args,
    )

    residual_stream = residual_stream_tracer(
        model=model,
        processor=processor,
        dataloader=dataloader,
        residual_stream_type=residual_stream_type,
        tokens_pooling_method=tokens_pooling_method,
        return_deepcopy=return_deepcopy,
    )

    entropy_type = (
        "prompt-entropy" if tokens_pooling_method is None else "dataset-entropy"
    )
    entropy = compute_layers_residual_stream_entropy(
        residual_stream, entropy_type=entropy_type
    )
    print(f"Entropy: {entropy}")
    print_heads_representations_info(residual_stream)

    del processor.tokenizer
    del model, processor, dataloader, residual_stream
    gc.collect()
    torch.cuda.empty_cache()


def test_residual_stream_output_layer_tracer_mean_pooling():
    test_residual_stream_template(
        residual_stream_type="output_layer", tokens_pooling_method="mean"
    )


def test_residual_stream_output_layer_tracer_none_pooling():
    test_residual_stream_template(
        residual_stream_type="output_layer",
        tokens_pooling_method=None,  # memory intensive
        return_deepcopy=False,
    )


def test_residual_stream_post_mlp_tracer_mean_pooling():
    test_residual_stream_template(
        residual_stream_type="post_mlp", tokens_pooling_method="mean"
    )


def test_residual_stream_post_mlp_tracer_none_pooling():
    test_residual_stream_template(
        residual_stream_type="post_mlp",
        tokens_pooling_method=None,  # memory intensive
        return_deepcopy=False,
    )


def test_residual_stream_heads_projection_tracer_mean_pooling():
    test_residual_stream_template(
        residual_stream_type="heads_projection", tokens_pooling_method="mean"
    )


if __name__ == "__main__":
    seed_all(42)

    # test_residual_stream_output_layer_tracer_mean_pooling()
    test_residual_stream_output_layer_tracer_none_pooling()
    # test_residual_stream_post_mlp_tracer_mean_pooling()
    # test_residual_stream_post_mlp_tracer_none_pooling()
    # test_residual_stream_heads_projection_tracer_mean_pooling()
