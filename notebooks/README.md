# Data Loading for Multimodal Finetuned Representations Analysis

This directory contains organized data loading utilities for your experimental results. You have two approaches to choose from based on your preferences.

## 🎯 Quick Start

**Option 1: Direct Python approach (Recommended for simplicity)**
```python
from data_loader import load_experimental_data

# Load all data with sensible defaults
loader = load_experimental_data()

# Access specific data
entropy_data = loader.get_model_entropy(
    dataset="cocoqa_img",
    model="llava_base",
    entropy_type="dataset_entropy", 
    stream_type="output_layer",
    pooling="last"
)
```

**Option 2: Configuration-based approach (Recommended for customization)**
```python
from config_data_loader import load_data_with_config

# Load using YAML configuration
loader = load_data_with_config("config.yaml")

# Same API, but uses config for paths and styling
entropy_data = loader.get_model_entropy(
    dataset="cocoqa_img",
    model="llava_base", 
    entropy_type="dataset_entropy",
    stream_type="output_layer",
    pooling="last"
)
```

## 📁 Data Organization

Your results are automatically organized as:

```
results/
├── cocoqa_img/                     # Image + text dataset
├── cocoqa_txt/                     # Text-only dataset
│   ├── models_data_measures/
│   │   ├── llava_base/            # Mapped from llava-1.5-7b-hf
│   │   └── llava_pretrained/      # Mapped from llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5
│   │       ├── dataset_entropy_output_layer_last.safetensors
│   │       ├── dataset_entropy_output_layer_mean.safetensors
│   │       ├── id_16_100_output_layer_last.safetensors
│   │       └── prompt_entropy_output_layer_none.safetensors
│   └── similarities/
│       ├── llava_base_vs_llava_pretrained_output_layer_last.safetensors
│       └── llava_base_vs_llava_pretrained_output_layer_mean.safetensors
└── transplanting_layers_benchmarking/
    ├── cocoqa_img/
    └── cocoqa_txt/
        ├── sliding_window_results.csv
        └── two_parts_results.csv
```

## 🔧 Key Features

### Clean Variable Naming
- Models: `llava_base`, `llava_pretrained` (instead of long HuggingFace names)
- Datasets: `cocoqa_img`, `cocoqa_txt`
- Stream types: `output_layer`, `post_mlp`, `heads_projection`
- Pooling: `last`, `mean`, `none`

### Relative Path Handling
- All paths are relative to the notebooks directory
- No hardcoded absolute paths
- Works regardless of where you clone the repository

### Multiple Data Access Patterns
```python
# Direct access
entropy = loader.get_model_entropy("cocoqa_img", "llava_base", "dataset_entropy", "output_layer", "last")

# Comparison helpers
from data_loader import get_entropy_comparison
entropy_comp = get_entropy_comparison(loader, "cocoqa_img", "output_layer", "last")

# Similarity measures
similarity = loader.get_similarity("cocoqa_img", "output_layer", "last", "linear_cka")

# Intrinsic dimension
id_data = loader.get_model_intrinsic_dimension("cocoqa_img", "llava_base", "output_layer", "last")
```

## 📊 Example Usage

### Basic Data Loading and Plotting
```python
# Load data
from data_loader import load_experimental_data
loader = load_experimental_data()

# Compare entropy across models
import matplotlib.pyplot as plt

entropy_comp = get_entropy_comparison(loader, "cocoqa_img", "output_layer", "last")
base_entropy = entropy_comp["llava_base"]["layers_dataset_entropy"]
pretrained_entropy = entropy_comp["llava_pretrained"]["layers_dataset_entropy"]

plt.figure(figsize=(10, 6))
plt.plot(base_entropy, label="LLaVA Base", marker='o')
plt.plot(pretrained_entropy, label="LLaVA Pretrained", marker='s')
plt.xlabel("Layer")
plt.ylabel("Entropy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Configuration-Based Plotting
```python
from config_data_loader import load_data_with_config, create_comparison_plot_with_config

loader = load_data_with_config()

# Automatically styled plots using config
configs = [
    ("output_layer", "last"),
    ("output_layer", "mean"),
    ("post_mlp", "last"), 
    ("post_mlp", "mean")
]

fig = create_comparison_plot_with_config(loader, "cocoqa_img", configs, "entropy_comparison.png")
```

## 🎨 Customization

### Approach 1: Direct Python (data_loader.py)
- Hardcoded sensible defaults
- Minimal configuration
- Fast to get started
- Best for: Quick analysis, standard workflows

### Approach 2: Configuration-based (config_data_loader.py + config.yaml)
- YAML-configurable paths and styling
- Easy to modify model names, colors, paths
- Reusable across different setups
- Best for: Custom setups, multiple environments, publication plots

Edit `config.yaml` to customize:
```yaml
# Custom model names
models:
  mapping:
    "llava-1.5-7b-hf": "baseline_model"
    "llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5": "finetuned_model"

# Custom plotting
plotting:
  colors:
    baseline_model: "#2E8B57"  # Sea green
    finetuned_model: "#DC143C"  # Crimson
```

## 🚀 Files Overview

- `data_loader.py` - Main data loading class with hardcoded sensible defaults
- `config_data_loader.py` - Configuration-based data loader using YAML
- `config.yaml` - Configuration file for paths, names, and plotting styles
- `example_data_loading.py` - Comprehensive examples and usage patterns
- `README.md` - This documentation

## 💡 Recommendations

1. **Start with `data_loader.py`** for quick analysis and exploration
2. **Switch to `config_data_loader.py`** when you need customization or publication plots
3. **Use the example files** to understand the different access patterns
4. **All data is automatically organized** with meaningful variable names

Both approaches provide the same API, so you can switch between them easily! 