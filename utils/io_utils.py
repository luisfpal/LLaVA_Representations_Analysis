import json
from typing import Dict, Optional
import torch
from safetensors.torch import save_file


def save_extracted_residual_stream_data(
    save_path: str, residual_data: Dict[str, Optional[object]]
) -> None:
    """
    Save extracted residual stream data and (optionally) prompts.

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

    save_file(tensor_dict, save_path)
    print(f"Saved tensor data to {save_path}")

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
