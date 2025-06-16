import os
from collections import defaultdict
from datasets import Dataset
from utils.dataset import get_hf_dataset_split

# Step 0: Setup
HF_CACHE_DIR = os.path.expanduser("~/scratch/huggingface/datasets")

# Step 1: Load datasets
karpathy_dataset = get_hf_dataset_split(
    dataset_path_or_name="yerevann/coco-karpathy",
    cache_dir=HF_CACHE_DIR,
    split="restval",
)

cocoqa_dataset = get_hf_dataset_split(
    dataset_path_or_name="ThucPD/coco-qa-vi",
    cache_dir=HF_CACHE_DIR,
    split="test",
)

print(f"\nKarpathy dataset size: {len(karpathy_dataset)}")
print(f"COCO QA VI dataset size: {len(cocoqa_dataset)}")

# Step 2: Extract shared image_ids
karpathy_ids = set(karpathy_dataset["filename"])
cocoqa_ids = set(cocoqa_dataset["image_id"])
shared_image_ids = karpathy_ids.intersection(cocoqa_ids)
print(f"\nUnique filenames in Karpathy dataset: {len(karpathy_ids)}")
print(f"Unique image IDs in COCO QA VI dataset: {len(cocoqa_ids)}")
print(f"\nShared image IDs: {len(shared_image_ids)}")

print("\n Filtering datasets with shared image IDs...")
# Step 3: Build image_id → captions map from Karpathy
karpathy_filtered = karpathy_dataset.filter(
    lambda x: x["filename"] in shared_image_ids and len(x["sentences"]) > 0,
    num_proc=8,
)

imageid_to_captions = {ex["filename"]: ex["sentences"] for ex in karpathy_filtered}

# Step 4: Build image_id → image (first one) and questions_answers list from COCOQA
qa_grouped = defaultdict(lambda: {"questions_answers": [], "image": None})

for ex in cocoqa_dataset:
    img_id = ex["image_id"]
    if img_id in shared_image_ids:
        qa_grouped[img_id]["questions_answers"].append(
            {"question": ex["question"], "answer": ex["answer"]}
        )
        if qa_grouped[img_id]["image"] is None:
            qa_grouped[img_id]["image"] = ex["image"]  # store first image occurrence

print("Building combined dataset...")
# Step 5: Merge into final dataset
final_data = []
for image_id, group in qa_grouped.items():
    captions = imageid_to_captions.get(image_id)
    if captions:
        final_data.append(
            {
                "image_id": image_id,
                "questions_answers": group["questions_answers"],
                "captions": captions,
                "image": group["image"],
            }
        )

print(f"\nFinal grouped dataset size: {len(final_data)}")

# Step 6: Convert to HuggingFace Dataset and shuffle
final_dataset = Dataset.from_list(final_data).shuffle(seed=42)

# Optional: Save to disk
output_dir = os.path.expanduser("~/scratch/datasets/cocoqa_captioning_restval")
final_dataset.save_to_disk(output_dir)
