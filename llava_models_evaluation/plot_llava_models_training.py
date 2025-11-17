import json
import matplotlib.pyplot as plt
from pathlib import Path

# Load the JSON files
data_dir = Path(__file__).parent

with open(data_dir / "Llava1.5-7B/Llava1.5-7B_benchmarks_results.json", 'r') as f:
    full_model_data = json.load(f)

with open(data_dir / "Llava1.5-7B-2-18/Llava1.5-7B-2-18_benchmarks_results.json", 'r') as f:
    model_2_18_data = json.load(f)

with open(data_dir / "Llava1.5-7B-18-32/Llava1.5-7B-18-32_benchmarks_results.json", 'r') as f:
    model_18_32_data = json.load(f)

# Extract data for Full Model (only non-empty data)
full_steps = []
full_txt_acc = []
full_img_acc = []
full_clips = []
full_cider = []

for key in sorted(full_model_data.keys(), key=lambda x: int(x.split('-')[-1])):
    step = int(key.split('-')[-1])
    # Only add data if the dictionaries are not empty
    if full_model_data[key]['cocoqa_txt'] and 'accuracy' in full_model_data[key]['cocoqa_txt']:
        full_steps.append(step)
        full_txt_acc.append(full_model_data[key]['cocoqa_txt']['accuracy'])
        full_img_acc.append(full_model_data[key]['cocoqa_img']['accuracy'])
        full_clips.append(full_model_data[key]['coco_captioning']['CLIP-S'])
        full_cider.append(full_model_data[key]['coco_captioning']['CIDEr'])

# Extract data for 2-18 Model (only non-empty data)
model_2_18_steps = []
model_2_18_txt_acc = []
model_2_18_img_acc = []
model_2_18_clips = []
model_2_18_cider = []

for key in sorted(model_2_18_data.keys(), key=lambda x: int(x.split('-')[-1])):
    step = int(key.split('-')[-1])
    # Only add data if the dictionaries are not empty
    if model_2_18_data[key]['cocoqa_txt'] and 'accuracy' in model_2_18_data[key]['cocoqa_txt']:
        model_2_18_steps.append(step)
        model_2_18_txt_acc.append(model_2_18_data[key]['cocoqa_txt']['accuracy'])
        model_2_18_img_acc.append(model_2_18_data[key]['cocoqa_img']['accuracy'])
        model_2_18_clips.append(model_2_18_data[key]['coco_captioning']['CLIP-S'])
        model_2_18_cider.append(model_2_18_data[key]['coco_captioning']['CIDEr'])

# Extract data for 18-32 Model (only non-empty data)
model_18_32_steps = []
model_18_32_txt_acc = []
model_18_32_img_acc = []
model_18_32_clips = []
model_18_32_cider = []

for key in sorted(model_18_32_data.keys(), key=lambda x: int(x.split('-')[-1])):
    step = int(key.split('-')[-1])
    # Only add data if the dictionaries are not empty
    if model_18_32_data[key]['cocoqa_txt'] and 'accuracy' in model_18_32_data[key]['cocoqa_txt']:
        model_18_32_steps.append(step)
        model_18_32_txt_acc.append(model_18_32_data[key]['cocoqa_txt']['accuracy'])
        model_18_32_img_acc.append(model_18_32_data[key]['cocoqa_img']['accuracy'])
        model_18_32_clips.append(model_18_32_data[key]['coco_captioning']['CLIP-S'])
        model_18_32_cider.append(model_18_32_data[key]['coco_captioning']['CIDEr'])

# Create 2x2 subplot figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Text VQA Accuracy
axes[0, 0].plot(full_steps, full_txt_acc, marker='o', label='Full', linewidth=2)
axes[0, 0].plot(model_2_18_steps, model_2_18_txt_acc, marker='s', label='2-18', linewidth=2)
axes[0, 0].plot(model_18_32_steps, model_18_32_txt_acc, marker='^', label='18-32', linewidth=2)
axes[0, 0].set_xlabel('Training Steps')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Text VQA Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))

# Plot 2: Multimodal VQA Accuracy
axes[0, 1].plot(full_steps, full_img_acc, marker='o', label='Full', linewidth=2)
axes[0, 1].plot(model_2_18_steps, model_2_18_img_acc, marker='s', label='2-18', linewidth=2)
axes[0, 1].plot(model_18_32_steps, model_18_32_img_acc, marker='^', label='18-32', linewidth=2)
axes[0, 1].set_xlabel('Training Steps')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Multimodal VQA Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))

# Share y-axis for accuracy plots
all_acc_values = full_txt_acc + model_2_18_txt_acc + full_img_acc + model_2_18_img_acc + model_18_32_txt_acc + model_18_32_img_acc
y_min, y_max = min(all_acc_values), max(all_acc_values)
y_range = y_max - y_min
axes[0, 0].set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
axes[0, 1].set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

# Plot 3: CLIP-S Score
axes[1, 0].plot(full_steps, full_clips, marker='o', label='Full', linewidth=2)
axes[1, 0].plot(model_2_18_steps, model_2_18_clips, marker='s', label='2-18', linewidth=2)
axes[1, 0].plot(model_18_32_steps, model_18_32_clips, marker='^', label='18-32', linewidth=2)
axes[1, 0].set_xlabel('Training Steps')
axes[1, 0].set_ylabel('CLIP-S Score')
axes[1, 0].set_title('CLIP-S Score')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))

# Plot 4: CIDEr Score
axes[1, 1].plot(full_steps, full_cider, marker='o', label='Full', linewidth=2)
axes[1, 1].plot(model_2_18_steps, model_2_18_cider, marker='s', label='2-18', linewidth=2)
axes[1, 1].plot(model_18_32_steps, model_18_32_cider, marker='^', label='18-32', linewidth=2)
axes[1, 1].set_xlabel('Training Steps')
axes[1, 1].set_ylabel('CIDEr Score')
axes[1, 1].set_title('CIDEr Score')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].ticklabel_format(style='scientific', axis='x', scilimits=(0,0))

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
output_path = data_dir / "training_metrics_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

plt.show()

