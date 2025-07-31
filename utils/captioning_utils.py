import os

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["PATH"] = (
    f"/usr/lib/jvm/java-8-openjdk-amd64/bin:{os.environ.get('PATH', '')}"
)

import torch
import contextlib
import json
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import tempfile
from typing import Dict, Optional, List
import gc
from torch.utils.data import DataLoader
from transformers.feature_extraction_utils import BatchFeature
from tqdm import tqdm
from .model_utils import ModelType, ProcessorType


def generate_captions(
    model: ModelType,
    processor: ProcessorType,
    tokenized_batch: BatchFeature,
    max_new_tokens: int = 50,
) -> List[str]:
    """Generate captions for a batch of images."""
    _, input_ids_length = tokenized_batch.input_ids.shape[:2]
    with torch.no_grad():
        kwargs_for_generate = {
            **tokenized_batch.to(model.device),
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }

        output_ids_tensor = model.generate(**kwargs_for_generate)

        # Extract only the generated tokens (excluding the prompt)
        generated_ids_only_tensor = output_ids_tensor[:, input_ids_length:]
        decoded_outputs_list = processor.batch_decode(
            generated_ids_only_tensor,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Clean up the generated captions
        predicted_captions = [
            decoded_output.strip() for decoded_output in decoded_outputs_list
        ]
    return predicted_captions


def save_captions_to_file(captions_data: List[Dict], file_path: str) -> None:
    """Save captions data to a JSON file with error handling."""
    try:
        with open(file_path, "w") as f:
            json.dump(captions_data, f, indent=2)
        print(f"Captions saved to: {file_path}")
    except Exception as e:
        print(f"Warning: Failed to save captions to {file_path}: {e}")


def benchmark_model_captioning_processed_dataloader(
    model: ModelType,
    processor: ProcessorType,
    processed_dataloader: DataLoader,
    max_new_tokens: int = 50,
    save_captions_path: Optional[str] = None,
) -> Dict[str, float]:
    """Benchmark model performance on captioning task with optional output saving."""
    
    print("\n+-+-+-🔤 Captioning dataset+-+-+-+\n")
    dataset_size = len(processed_dataloader.dataset)
    progress_bar = tqdm(total=dataset_size, desc="Processing samples", unit="sample")
    update_every = max(1, int(0.1 * dataset_size))
    last_update = 0

    # Initialize data structures for evaluation
    generated_captions = [None] * dataset_size
    ground_truth_images = [None] * dataset_size
    ground_truth_annotations = []
    annotation_id = 0
    sample_idx = 0
    
    # Initialize captions for saving if requested
    captions_to_save = []
    if save_captions_path:
        captions_to_save = [None] * dataset_size

    # Process each batch
    for (
        tokenized_batch,
        reference_captions_batch,
        image_ids_batch,
    ) in processed_dataloader:
        predicted_captions = generate_captions(
            model, processor, tokenized_batch, max_new_tokens
        )
        current_batch_size = len(predicted_captions)

        for batch_sample_idx in range(current_batch_size):
            # Use real image ID if available and valid, otherwise use sample index
            raw_image_id = (
                image_ids_batch[batch_sample_idx]
                if batch_sample_idx < len(image_ids_batch)
                else None
            )
            real_image_id = raw_image_id if raw_image_id is not None else sample_idx

            # Store generated caption for evaluation
            generated_captions[sample_idx] = {
                "image_id": real_image_id,
                "caption": predicted_captions[batch_sample_idx],
            }
            
            # Store caption for saving if requested
            if save_captions_path:
                captions_to_save[sample_idx] = {
                    "image_id": real_image_id,
                    "caption": predicted_captions[batch_sample_idx],
                }

            # Add image to ground truth data
            ground_truth_images[sample_idx] = {"id": real_image_id}

            # Store all reference captions for this image
            for reference_caption in reference_captions_batch[batch_sample_idx]:
                ground_truth_annotations.append(
                    {
                        "image_id": real_image_id,
                        "id": annotation_id,
                        "caption": reference_caption,
                    }
                )
                annotation_id += 1
            
            sample_idx += 1
        
        # Update progress bar
        if sample_idx % update_every == 0 or sample_idx == dataset_size:
            progress_bar.update(sample_idx - last_update)
            last_update = sample_idx
    
    progress_bar.close()
    torch.cuda.empty_cache()
    gc.collect()

    # Save captions if requested
    if save_captions_path:
        save_captions_to_file(captions_to_save, save_captions_path)

    # Prepare COCO evaluation data
    ground_truth_coco_format = {
        "info": {},
        "licenses": [],
        "type": "captions",
        "images": ground_truth_images,
        "annotations": ground_truth_annotations,
    }
    
    # Create temporary files for COCO evaluation
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f_gt:
        json.dump(ground_truth_coco_format, f_gt)
        ground_truth_path = f_gt.name

    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f_res:
        json.dump(generated_captions, f_res)
        results_path = f_res.name

    # Evaluate captions using COCO metrics
    print("\n+-+-+-📝 Evaluating captions+-+-+-+\n")
    with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull):
        coco = COCO(ground_truth_path)
        coco_result = coco.loadRes(results_path)
        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.params["image_id"] = coco_result.getImgIds()
        coco_eval.evaluate()
    
    return {
        "BLEU-1": coco_eval.eval["Bleu_1"],
        "BLEU-2": coco_eval.eval["Bleu_2"],
        "BLEU-3": coco_eval.eval["Bleu_3"],
        "BLEU-4": coco_eval.eval["Bleu_4"],
        "METEOR": coco_eval.eval["METEOR"],
        "ROUGE-L": coco_eval.eval["ROUGE_L"],
        "CIDEr": coco_eval.eval["CIDEr"],
        "SPICE": coco_eval.eval["SPICE"],
    }
