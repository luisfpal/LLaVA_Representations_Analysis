import torch
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


def main():
    seed = 42
    batch_size = 25
    dataset_path_or_name = "~/scratch/datasets/cocoqa_unified"

    multimodal_model, processor = setup_multimodal_model(
        multimodal_model_name_or_path="llava-hf/llava-1.5-7b-hf",
        model_cache_dir="~/scratch/huggingface/hub",
        language_model_name_or_path="lmsys/vicuna-7b-v1.5",
        pretrained_projector_name_or_path="liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5",
        replace_pretrained_projector=False,
        replace_language_model=False,
        device_map="cpu",
        attn_implementation="sdpa",
        model_dtype=torch.float32,
    )
    multimodal_model_pretrained_connector = setup_multimodal_model(
        multimodal_model_name_or_path="llava-hf/llava-1.5-7b-hf",
        model_cache_dir="~/scratch/huggingface/hub",
        language_model_name_or_path="lmsys/vicuna-7b-v1.5",
        pretrained_projector_name_or_path="liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5",
        replace_pretrained_projector=True,
        replace_language_model=True,
        device_map="cpu",
        attn_implementation="sdpa",
        model_dtype=torch.float32,
        skip_processor=True,
    )
    model = transplant_layers_weights(
        source_model=multimodal_model,
        target_model=multimodal_model_pretrained_connector,
        method="layers_list",
        layers=[32],
        in_place_transplantation=True,
    )

    processed_dataloader = get_dataloader(
        **{
            **DATASETS["cocoqa_txt"],
            "dataset_path_or_name": dataset_path_or_name,
            "seed": seed,
            "processor": processor,
            "batch_size": batch_size,
            "answer_letters_with_processed_batch": True,
        },
    )

    model.to(device="cuda:0", dtype=torch.float16)
    results = benchmark_model_vqa_processed_dataloader(
        model=model,
        processed_dataloader=processed_dataloader,
        processor=processor,
        max_new_tokens=10,
    )
    print(results)


if __name__ == "__main__":
    main()
