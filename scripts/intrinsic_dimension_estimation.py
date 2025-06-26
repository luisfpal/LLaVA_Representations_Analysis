import os
import torch
import argparse
import pandas as pd
from typing import Dict, Optional
from utils import (
    seed_all,
    sample_unique_row_indices,
    compute_intrinsic_dimension,
    load_layers_residual_stream,
)


def analyze_intrinsic_dimension(
    residual_stream_data_path: str,
    algorithm: str = "twoNN",
    k: Optional[int] = None,
    fraction: Optional[float] = None,
    downsample_size: Optional[int] = None,
    full_output: bool = False,
    range_max: Optional[int] = None,
) -> Dict[int, float]:
    """
    Compute intrinsic dimension for each layer of a model.

    Args:
        residual_stream_data_path: Path to the residual stream file
        algorithm: Algorithm to use for ID estimation ('twoNN', 'MLE', or 'scaling_gride')
        k: Number of nearest neighbors for MLE/scaling_gride algorithm
        fraction: Fraction of points for twoNN algorithm
        downsample_size: Maximum number of samples to consider
        full_output: Whether to return full output or just mean for MLE
        range_max: Maximum neighbor rank to consider for scaling_gride algorithm

    Returns:
        Dictionary mapping layer indices to intrinsic dimension scores
    """
    # Load data
    residual_stream_data_path = os.path.expanduser(residual_stream_data_path)
    is_multi_layer, data = load_layers_residual_stream(residual_stream_data_path)

    intrinsic_dimensions = {}
    reduced_unique_sample_indices = None
    data_size = None

    if is_multi_layer:
        # Multi-layer case
        layers = sorted(data.keys())
        print(f"Found {len(layers)} layers: {layers}")

        # Compute intrinsic dimension for each layer
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

            intrinsic_dim = compute_intrinsic_dimension(
                R,
                algorithm=algorithm,
                k=k,
                fraction=fraction,
                full_output=full_output,
                range_max=range_max,
            )
            intrinsic_dimensions[layer_idx] = intrinsic_dim
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

        intrinsic_dim = compute_intrinsic_dimension(
            R,
            algorithm=algorithm,
            k=k,
            fraction=fraction,
            full_output=full_output,
            range_max=range_max,
        )
        intrinsic_dimensions[-1] = intrinsic_dim  # Use -1 to indicate the default layer

    return intrinsic_dimensions


def create_filename_from_path(path: str) -> str:
    subpath = path.split("representations", 1)[1].split(".safetensors")[0]
    subpath = subpath.replace("layers_representations/", "")
    components = subpath.strip("/").split("/")
    filename = "_".join(components)
    return filename.lower()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze intrinsic dimension of residual stream representations"
    )
    parser.add_argument("--residual-stream-path", type=str, required=True)
    parser.add_argument("--result-parent-dir", type=str, required=True)
    parser.add_argument(
        "--algorithm",
        type=str,
        default="twoNN",
        choices=["twoNN", "MLE", "scaling_gride"],
    )
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--fraction", type=float, default=None)
    parser.add_argument("--downsample-size", type=int, default=None)
    parser.add_argument("--full-output", action="store_true")
    parser.add_argument("--range-max", type=int, default=None)
    args = parser.parse_args()

    # Set random seed for reproducibility
    seed_all(42)

    # Set up directories for results
    base_dir = os.path.expanduser(args.result_parent_dir)
    results_dir = os.path.join(
        base_dir, f"results/intrinsic_dimension/{args.algorithm}"
    )

    # Create directories if they don't exist
    os.makedirs(results_dir, exist_ok=True)

    # Filename from residual stream path
    filename = create_filename_from_path(args.residual_stream_path)
    if args.downsample_size is not None:
        filename += f"_reps-ds{args.downsample_size}"
    if args.algorithm == "MLE":
        if args.k is not None:
            filename += f"_k-{args.k}"
        if args.full_output:
            filename += "_full-out"
    if args.algorithm == "twoNN":
        if args.fraction is not None:
            filename += f"_frac-{args.fraction}"
    if args.algorithm == "scaling_gride":
        if args.k is not None:
            filename += f"_k-{args.k}"
        if args.range_max is not None:
            filename += f"_range_max-{args.range_max}"
    csv_path = os.path.join(results_dir, f"{filename}.csv")

    if not os.path.exists(csv_path):
        # Compute intrinsic dimension
        intrinsic_dimensions = analyze_intrinsic_dimension(
            residual_stream_data_path=args.residual_stream_path,
            algorithm=args.algorithm,
            k=args.k,
            fraction=args.fraction,
            downsample_size=args.downsample_size,
            full_output=args.full_output,
            range_max=args.range_max,
        )

        # Save results to CSV file
        intrinsic_dimension_df = pd.DataFrame(
            list(intrinsic_dimensions.items()),
            columns=["layer_index", "intrinsic_dimension"],
        )

        intrinsic_dimension_df.to_csv(csv_path, index=False)
        print(f"Layer-wise intrinsic dimensions saved to {csv_path}")

    else:
        print(f"Layer-wise intrinsic dimensions already computed: {csv_path}")


if __name__ == "__main__":
    main()
