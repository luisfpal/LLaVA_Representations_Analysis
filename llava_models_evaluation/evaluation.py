import os
import json
import argparse
from pathlib import Path

import torch
from rich import print

from utils import (
    load_hf_model_and_processor_or_tokenizer,
    get_dataloader_for_captioning,
    get_dataloader,
    benchmark_model_captioning_processed_dataloader,
    benchmark_model_vqa_processed_dataloader,
    setup_multimodal_model,
)
from src.transplantation_benchmarking_common import create_benchmark_model_wrapper


# ===== Configuration =====
FILE_PARENT_DIR = Path(__file__).parent
MODEL_CACHE_DIR = os.path.expanduser("~/scratch/huggingface/hub")
DATASET_PATH = os.path.expanduser("~/scratch/datasets/cocoqa_unified")

# Evaluation parameters
SEED = 42
BATCH_SIZE = 10
MAX_NEW_TOKENS = 100

# CLIP model arguments for benchmarking
CLIP_ARGS = argparse.Namespace(
    clip_model_name_or_path="openai/clip-vit-large-patch14-336",
    clip_cache_dir=MODEL_CACHE_DIR,
    clip_weight=2.5,
    clip_batch_size=16
)

# Dataset configurations
COCOQA_COMMON_CONFIG = {
    "guide_text": "Answer the question using a single word or phrase.\n",
    "downsample_size": 2500,
    "answer_letters_with_processed_batch": True,
}

DATASETS = {
    "cocoqa_txt": {
        "texts_qa": True,
        **COCOQA_COMMON_CONFIG,
    },
    "cocoqa_img": {
        "images_qa": True,
        **COCOQA_COMMON_CONFIG,
    },
    "coco_captioning": {
        "downsample_size": 2500,
        "return_captions": True,
        "return_image_ids": True,
        "return_images": True,
    },
}

# Model collections to evaluate
MODELS_COLLECTION = {
    # "Llava1.5-7B": "lbasile/llava-",
    # "Llava1.5-7B-2-18": "lbasile/llava_2_18-",
    # "Llava1.5-7B-18-32": "lbasile/llava_18_32-",
    "Llava1.5-7B-2-12": "lbasile/llava_2_12-",
    # "Llava1.5-7B-frozen-18-32": "lbasile/llava_frozen_0_32-",
}


# ===== Helper Functions =====
def get_pretrained_llava_model():
    """
    Load the pretrained LLava model with custom projector and language model.
    
    Returns:
        Tuple of (model, processor) for the pretrained LLava model.
    """
    model, processor = setup_multimodal_model(
        multimodal_model_name_or_path="llava-hf/llava-1.5-7b-hf",
        model_cache_dir=MODEL_CACHE_DIR,
        replace_pretrained_projector=True,
        pretrained_projector_name_or_path="liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5",
        replace_language_model=True,
        language_model_name_or_path="lmsys/vicuna-7b-v1.5",
        skip_processor=False,
        device_map="cpu",
        attn_implementation="flash_attention_2",
        model_dtype=torch.float16,
    )
    return model, processor


def fix_processor_patch_size(processor):
    """
    Ensure processor has patch_size attribute for compatibility.
    
    Args:
        processor: The model processor to fix.
    """
    if not hasattr(processor, "image_processor"):
        return
    
    image_processor = processor.image_processor
    
    if hasattr(image_processor, "patch_size"):
        processor.patch_size = image_processor.patch_size
    elif hasattr(image_processor, "size"):
        # Fallback: infer patch_size from size config
        processor.patch_size = image_processor.size.get("patch_size", 14)



# ===== Main Execution =====
def main(collection_name):
    
    print(f"[bold green]Evaluating models from {collection_name} collection[/bold green]")
    
    models_collection_prefix = MODELS_COLLECTION[collection_name]
    # model_names = [f"{models_collection_prefix}{i}" for i in range(1000, 10001, 1000)]
    model_names = [f"{models_collection_prefix}0"]

    # Setup output directory
    output_dir = FILE_PARENT_DIR / collection_name
    os.makedirs(output_dir, exist_ok=True)
    results_path = output_dir / f"{collection_name}_benchmarks_results.json"

    # Load existing results or initialize new results dictionary
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)
        print(f"Loaded existing results from {results_path}")
    else:
        results = {
            model_name: {dataset_name: {} for dataset_name in DATASETS.keys()}
            for model_name in model_names
        }

    # Evaluate each model
    for model_name in model_names:
        # Skip models with complete results
        if model_name in results and all(results[model_name].get(ds) for ds in DATASETS.keys()):
            print(f"\n[yellow]Skipping {model_name} - already completed[/yellow]")
            continue
        
        print(f"\n[bold green]Processing {model_name}[/bold green]")
        
        # Initialize results structure for new models
        if model_name not in results:
            results[model_name] = {dataset_name: {} for dataset_name in DATASETS.keys()}
        
        # Load model and processor (use pretrained for model 0)
        if "-0" in model_name:
            model, processor = get_pretrained_llava_model()
        else:
            model, processor = load_hf_model_and_processor_or_tokenizer(
                model_name,
                MODEL_CACHE_DIR,
            )
        
        # Fix processor patch_size attribute
        fix_processor_patch_size(processor)
        
        # Evaluate on each dataset
        for dataset_name in DATASETS.keys():
            print(f"  Evaluating on {dataset_name}...")
            
            # Select appropriate dataloader and benchmark function
            if dataset_name == "coco_captioning":
                dataloader_func = get_dataloader_for_captioning
                benchmark_func = benchmark_model_captioning_processed_dataloader
            else:
                dataloader_func = get_dataloader
                benchmark_func = benchmark_model_vqa_processed_dataloader
            
            # Create benchmark wrapper
            benchmark_model = create_benchmark_model_wrapper(benchmark_func, CLIP_ARGS)
            
            # Prepare dataloader
            dataloader = dataloader_func(
                **DATASETS[dataset_name],
                dataset_path_or_name=DATASET_PATH,
                seed=SEED,
                processor=processor,
                batch_size=BATCH_SIZE,
            )
            
            # Run benchmark and save results
            output_path = output_dir / f"{model_name.replace('/', '-')}_{dataset_name}.json"
            results[model_name][dataset_name] = benchmark_model(
                model,
                dataloader,
                processor,
                max_new_tokens=MAX_NEW_TOKENS,
                save_outputs_path=str(output_path),
            )
            
            print(f"  Results: {results[model_name][dataset_name]}")
        
        # Save aggregated results after each model
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    print(f"[bold green]All models processed and results saved to {results_path}[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection_name", type=str, default="Llava1.5-7B")
    args = parser.parse_args()
    main(args.collection_name)
