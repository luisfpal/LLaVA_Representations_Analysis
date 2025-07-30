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
from typing import Dict
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
) -> Dict[str, float]:
    _, input_ids_length = tokenized_batch.input_ids.shape[:2]
    with torch.no_grad():
        kwargs_for_generate = {
            **tokenized_batch.to(model.device),
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }

        output_ids_tensor = model.generate(
            **kwargs_for_generate,
        )
        # Slice to get only the generated tokens (excluding the prompt)
        generated_ids_only_tensor = output_ids_tensor[:, input_ids_length:]
        decoded_outputs_list = processor.batch_decode(
            generated_ids_only_tensor,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Decode the outputs and remove leading/trailing whitespace
        predicted_captions = [
            decoded_output.strip() for decoded_output in decoded_outputs_list
        ]
    return predicted_captions


def benchmark_model_captioning_processed_dataloader(
    model: ModelType,
    processor: ProcessorType,
    processed_dataloader: DataLoader,
    max_new_tokens: int = 50,
) -> Dict[str, float]:
    
    print("\n+-+-+-🔤 Captioning dataset+-+-+-+\n")
    dataset_size = len(processed_dataloader.dataset)
    progress_bar = tqdm(total=dataset_size, desc="Processing samples", unit="sample")
    update_every = max(1, int(0.1 * dataset_size))
    last_update = 0

    # Initialize lists to store generated captions and ground truth data
    # Needed for the COCO evaluation library
    # Pre-allocate for performance since we know the exact size
    generated_results = [None] * dataset_size
    gt_images = [None] * dataset_size
    gt_annotations = []  # Variable size due to multiple captions per image
    annotation_id = 0
    dummy_image_id = 0

    for tokenized_batch, ref_captions_batch in processed_dataloader:
        generated_texts = generate_captions(model, processor, tokenized_batch, max_new_tokens)

        current_batch_size = len(generated_texts)
        
        for sample_idx in range(current_batch_size):
            # Save generated caption
            generated_results[dummy_image_id] = {
                "image_id": dummy_image_id,
                "caption": generated_texts[sample_idx],
            }

            # Add image to ground truth images list 
            gt_images[dummy_image_id] = {
                "id": dummy_image_id,
            }

            # Save all GT captions for this image (one by one as required)
            for ref in ref_captions_batch[sample_idx]:
                gt_annotations.append({
                    "image_id": dummy_image_id,
                    "id": annotation_id,
                    "caption": ref,
                })
                annotation_id += 1
            
            dummy_image_id += 1  # Increment after processing this sample
        
        if dummy_image_id % update_every == 0 or dummy_image_id == dataset_size:
            progress_bar.update(dummy_image_id - last_update)
            last_update = dummy_image_id
    
    progress_bar.close()
    torch.cuda.empty_cache()
    gc.collect()

    # Create temporary reference file in COCO format
    gt_coco_format = {
        # !if these fields are needed due to runtime errors,
        # they can be found in utils.constants.py
        "info": {},
        "licenses": [],
        "type": "captions",
        "images": gt_images,
        "annotations": gt_annotations,
    }
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f_gt:
        json.dump(gt_coco_format, f_gt)
        ref_path = f_gt.name

    # Create temporary result file - just the annotations list as expected by loadRes
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f_res:
        json.dump(generated_results, f_res)
        res_path = f_res.name

    # Evaluate silently
    print("\n+-+-+-📝 Evaluating captions+-+-+-+\n")
    with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull):
        coco = COCO(ref_path)
        coco_result = coco.loadRes(res_path)
        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.params["image_id"] = coco_result.getImgIds()
        coco_eval.evaluate()
    
    return {
        "BLEU-1": coco_eval.eval['Bleu_1'],
        "BLEU-2": coco_eval.eval['Bleu_2'],
        "BLEU-3": coco_eval.eval['Bleu_3'],
        "BLEU-4": coco_eval.eval['Bleu_4'],
        "METEOR": coco_eval.eval['METEOR'],
        "ROUGE-L": coco_eval.eval['ROUGE_L'],
        "CIDEr": coco_eval.eval['CIDEr'],
        "SPICE": coco_eval.eval['SPICE'],
    }
