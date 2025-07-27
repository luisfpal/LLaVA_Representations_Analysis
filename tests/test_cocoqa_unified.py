from datasets import load_from_disk
import os
from rich import print

dataset_dir = os.path.expanduser("~/scratch/datasets/cocoqa_unified")

loaded_dataset = load_from_disk(dataset_dir)

# Size of the dataset
print(f"Dataset size: {len(loaded_dataset)} samples")

features_names = loaded_dataset.column_names
print(f"Features in the dataset: {features_names}")

# Get a subset of 2500 samples
subset_size = 2500
subset_dataset = loaded_dataset.shuffle(seed=42).select(range(subset_size))

# Size of subset dataset
print(f"Subset dataset size: {len(subset_dataset)} samples")

# Display the first few samples
for i in range(3):
    print(f"Sample {i + 1}:")
    print(subset_dataset[i])
    print("-" * 40)
