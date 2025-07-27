import os
import torch
import argparse
import pandas as pd
from typing import Dict, Optional
from utils import (
    seed_all,
    sample_unique_row_indices,
    compute_matrix_renyi_entropy,
    load_layers_residual_stream,
)


def analyze_renyi_entropy(
    residual_stream_data_path: str,
    alpha: float = 1.0,
    downsample_size: Optional[int] = None,
) -> Dict[int, float]:
    """
    Compute Rényi entropy for each layer of a model.

    Args:
        residual_stream_data_path: Path to the residual stream file
        alpha: Rényi entropy order (α = 1 for Shannon entropy)
        downsample_size: Maximum number of samples to consider

    Returns:
        Dictionary mapping layer indices to Rényi entropy scores
    """
    # Load data
    residual_stream_data_path = os.path.expanduser(residual_stream_data_path)
    is_multi_layer, data = load_layers_residual_stream(residual_stream_data_path)

    renyi_entropies = {}
    reduced_unique_sample_indices = None
    data_size = None

    if is_multi_layer:
        # Multi-layer case
        layers = sorted(data.keys())
        print(f"Found {len(layers)} layers: {layers}")

        # Compute Rényi entropy for each layer
        for idx, layer_idx in enumerate(layers):
            R = data[layer_idx]

            if idx == 0:
                data_size = R.shape[0]
                if downsample_size is not None and data_size > downsample_size:
                    # Downsample the first layer to reduce computation
                    reduced_unique_sample_indices = sample_unique_row_indices(
                        R, downsample_size
                    )
                    print(f"Downsampling from {data_size} to {downsample_size} samples")

            if reduced_unique_sample_indices is not None:
                R = R[reduced_unique_sample_indices]

            # Convert to tensor if needed
            if not isinstance(R, torch.Tensor):
                R = torch.tensor(R, dtype=torch.float32)

            renyi_entropy = compute_matrix_renyi_entropy(R, alpha=alpha)
            renyi_entropies[layer_idx] = renyi_entropy
    else:
        # Single layer case
        R = data
        data_size = R.shape[0]

        if downsample_size is not None and data_size > downsample_size:
            # Downsample the data to reduce computation
            reduced_unique_sample_indices = sample_unique_row_indices(
                R, downsample_size
            )
            print(f"Downsampling from {data_size} to {downsample_size} samples")
            R = R[reduced_unique_sample_indices]

        # Convert to tensor if needed
        if not isinstance(R, torch.Tensor):
            R = torch.tensor(R, dtype=torch.float32)

        renyi_entropy = compute_matrix_renyi_entropy(R, alpha=alpha)
        renyi_entropies[-1] = renyi_entropy  # Use -1 to indicate the default layer

    return renyi_entropies


def create_filename_from_path(path: str) -> str:
    subpath = path.split("representations", 1)[1].split(".safetensors")[0]
    subpath = subpath.replace("layers_representations/", "")
    components = subpath.strip("/").split("/")
    filename = "_".join(components)
    return filename.lower()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Rényi entropy of residual stream representations"
    )
    parser.add_argument("--residual-stream-path", type=str, required=True)
    parser.add_argument("--result-parent-dir", type=str, required=True)
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Rényi entropy order (α = 1 for Shannon entropy)",
    )
    parser.add_argument("--downsample-size", type=int, default=None)
    args = parser.parse_args()

    # Set random seed for reproducibility
    seed_all(42)

    # Set up directories for results
    base_dir = os.path.expanduser(args.result_parent_dir)
    results_dir = os.path.join(base_dir, "results/renyi_entropy")

    # Create directories if they don't exist
    os.makedirs(results_dir, exist_ok=True)

    # Filename from residual stream path
    filename = create_filename_from_path(args.residual_stream_path)
    if args.downsample_size is not None:
        filename += f"_reps-ds{args.downsample_size}"

    # Add alpha parameter to filename
    if args.alpha == 1.0:
        filename += "_shannon"  # Shannon entropy (α=1)
    else:
        filename += f"_alpha-{args.alpha}"

    csv_path = os.path.join(results_dir, f"{filename}.csv")

    if not os.path.exists(csv_path):
        # Compute Rényi entropy
        renyi_entropies = analyze_renyi_entropy(
            residual_stream_data_path=args.residual_stream_path,
            alpha=args.alpha,
            downsample_size=args.downsample_size,
        )

        # Save results to CSV file
        renyi_entropy_df = pd.DataFrame(
            list(renyi_entropies.items()),
            columns=["layer_index", "renyi_entropy"],
        )

        renyi_entropy_df.to_csv(csv_path, index=False)
        print(f"Layer-wise Rényi entropies saved to {csv_path}")

    else:
        print(f"Layer-wise Rényi entropies already computed: {csv_path}")


if __name__ == "__main__":
    main()
