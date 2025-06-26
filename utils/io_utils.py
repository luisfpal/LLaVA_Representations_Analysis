import json
from typing import Dict, Optional, Tuple
import torch
import gc
from safetensors.torch import save_file
from safetensors.torch import load_file
import numpy as np


def save_extracted_residual_stream_data(
    save_path: str, residual_data: Dict[str, Optional[object]]
) -> None:
    """
    Save extracted residual stream data and (optionally) prompts.
    Optimized for large tensor dictionaries with memory management.

    Args:
        save_path: Path to save the tensor data (as .safetensors).
        residual_data: Dictionary with keys:
            - "residual_stream_data": Dict[str, torch.Tensor] (required)
            - "prompts": List[str] (optional)
    """
    tensor_dict = residual_data.get("residual_stream_data")
    if not isinstance(tensor_dict, dict) or not all(
        isinstance(t, torch.Tensor) for t in tensor_dict.values()
    ):
        raise ValueError(
            "'residual_stream_data' must be a dictionary of torch.Tensor values."
        )

    # OPTIMIZATION 1: Ensure all tensors are on CPU and contiguous for faster I/O
    print("Preparing tensors for saving...")
    optimized_tensor_dict = {}

    for key, tensor in tensor_dict.items():
        # Move to CPU if needed and ensure contiguous memory layout
        if tensor.is_cuda:
            cpu_tensor = tensor.cpu()
        else:
            cpu_tensor = tensor

        # Ensure contiguous memory layout for faster I/O
        if not cpu_tensor.is_contiguous():
            cpu_tensor = cpu_tensor.contiguous()

        optimized_tensor_dict[key] = cpu_tensor

    # OPTIMIZATION 2: Clear CUDA cache before saving
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # OPTIMIZATION 3: Use safetensors with explicit memory management
    print(f"Saving {len(optimized_tensor_dict)} tensors...")
    save_file(optimized_tensor_dict, save_path)
    print(f"Saved tensor data to {save_path}")

    # OPTIMIZATION 4: Clean up optimized tensors
    del optimized_tensor_dict
    gc.collect()

    # Handle prompts if provided
    prompts = residual_data.get("prompts")
    if prompts is not None:
        if not isinstance(prompts, list) or not all(
            isinstance(p, str) for p in prompts
        ):
            raise ValueError("'prompts' must be a list of strings.")
        prompts_path = save_path.replace(".safetensors", "_prompts.json")
        with open(prompts_path, "w") as f:
            json.dump(prompts, f, indent=2)
        print(f"Saved prompts to {prompts_path}")


def load_layers_residual_stream(file_path: str) -> Tuple[bool, Dict[int, np.ndarray]]:
    """Load residual stream data from a saved safetensors file."""
    print(f"Loading data from {file_path}")

    # Load tensors from safetensors file
    tensors_dict = load_file(file_path)

    # Check if we have multiple layers or just one
    multi_layer = sum([key.startswith("layer_") for key in tensors_dict.keys()]) > 1

    if multi_layer:
        # Parse layer indices from keys
        layers_data = {}
        for key, tensor in tensors_dict.items():
            if key.startswith("layer_"):
                layer_idx = int(key.split("_")[1])
                data = (
                    tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
                )
                layers_data[layer_idx] = data

        return True, layers_data
    else:
        # Single layer case
        tensor = list(tensors_dict.values())[0]
        data = tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
        return False, data
