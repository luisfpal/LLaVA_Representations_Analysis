import os
import torch
import argparse
import matplotlib.pyplot as plt
from safetensors.torch import load_file
import pandas as pd
import numpy as np
from utils import compute_neighborhood_overlap, seed_all, sample_unique_row_indices


def load_residual_stream(file_path):
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


def plot_layer_overlaps(
    layer_overlaps_df,
    save_path,
    plot_title="Layer-wise Neighborhood Overlap",
):
    """
    Create a plot of layer-wise neighborhood overlaps from a DataFrame.

    Args:
        layer_overlaps_df: DataFrame with columns ["Layer", "Neighborhood Overlap"]
        save_path: Path to save the plot
        plot_title: Title for the plot
    """
    # Ensure the data is sorted by the Layer index
    df = layer_overlaps_df.sort_values(by="Layer").reset_index(drop=True)

    layers = df["Layer"].values
    overlap_values = df["Neighborhood Overlap"].values

    # Create the plot with good aesthetics
    plt.figure(figsize=(10, 6))
    plt.plot(
        layers,
        overlap_values,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=8,
        color="#2076B2",
        markerfacecolor="white",
        markeredgewidth=2,
    )

    # Add grid, labels, and title
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel("Layer Index", fontsize=14)
    plt.ylabel("Neighborhood Overlap", fontsize=14)
    plt.title(plot_title, fontsize=16)

    # Set axis limits with a bit of padding
    plt.ylim([max(0, min(overlap_values) - 0.05), 1.05])

    # Add background color and style
    plt.gca().set_facecolor("#F8F8F8")
    plt.gcf().set_facecolor("white")

    # Customize x-ticks: select 10 evenly spaced ticks
    num_ticks = 10
    if len(layers) <= num_ticks:
        tick_layers = layers
    else:
        indices = np.linspace(0, len(layers) - 1, num=num_ticks, dtype=int)
        tick_layers = layers[indices]

    plt.xticks(tick_layers, fontsize=14)
    plt.yticks(fontsize=14)

    # Finalize layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {save_path}")


def analyze_layers(
    residual_stream_data1_path,
    residual_stream_data2_path,
    maxk=30,
    downsample_size=None,
):
    """
    Analyze neighborhood overlap between models layer by layer.

    Args:
        residual_stream_data1_path: Path to the first model's residual stream file
        residual_stream_data2_path: Path to the second model's residual stream file
        maxk: Maximum number of nearest neighbors to consider

    Returns:
        Dictionary mapping layer indices to overlap scores
    """
    # Load data from both models
    residual_stream_data1_path = os.path.expanduser(residual_stream_data1_path)
    residual_stream_data2_path = os.path.expanduser(residual_stream_data2_path)
    is_multi_layer1, data1 = load_residual_stream(residual_stream_data1_path)
    is_multi_layer2, data2 = load_residual_stream(residual_stream_data2_path)

    # Ensure both have the same format (multi-layer or single-layer)
    if is_multi_layer1 != is_multi_layer2:
        raise ValueError(
            "Both models must have the same layer structure (single or multi-layer)"
        )

    layer_overlaps = {}
    reduced_unique_sample_indices = None
    data_size = None
    if is_multi_layer1:
        # Multi-layer case
        # Find common layers between the two models
        common_layers = sorted(set(data1.keys()).intersection(set(data2.keys())))

        if not common_layers:
            raise ValueError("No common layers found between the two models")

        print(f"Found {len(common_layers)} common layers: {common_layers}")

        # Compute overlap for each common layer
        for idx, layer_idx in enumerate(common_layers):
            print(f"\nAnalyzing layer {layer_idx}...")
            R1 = data1[layer_idx]
            R2 = data2[layer_idx]
            if idx == 0:
                data_size = R1.shape[0]
                if downsample_size is not None and data_size > downsample_size:
                    # Downsample the first layer to reduce computation
                    reduced_unique_sample_indices = sample_unique_row_indices(
                        R1, downsample_size
                    )
                    print(f"Downsampling from {data_size} to {downsample_size} samples")
            if reduced_unique_sample_indices is not None:
                R1 = R1[reduced_unique_sample_indices]
                R2 = R2[reduced_unique_sample_indices]
            overlap = compute_neighborhood_overlap(R1, R2, maxk=maxk)
            print(f"Layer {layer_idx} overlap: {overlap:.4f}")
            layer_overlaps[layer_idx] = overlap
    else:
        # Single layer case
        print("Analyzing single layer...")
        R1 = data1
        R2 = data2
        data_size = R1.shape[0]
        if downsample_size is not None and data_size > downsample_size:
            # Downsample the data to reduce computation
            reduced_unique_sample_indices = sample_unique_row_indices(
                R1, downsample_size
            )
            print(f"Downsampling from {data_size} to {downsample_size} samples")
            R1 = R1[reduced_unique_sample_indices]
            R2 = R2[reduced_unique_sample_indices]
        overlap = compute_neighborhood_overlap(R1, R2, maxk=maxk)
        layer_overlaps[-1] = overlap  # Use -1 to indicate the default layer
        print(f"Single layer overlap: {overlap:.4f}")

    return layer_overlaps


def main():
    parser = argparse.ArgumentParser(
        description="Analyze residual stream neighborhood overlap"
    )
    parser.add_argument("--residual-stream-path1", type=str, required=True)
    parser.add_argument("--residual-stream-path2", type=str, required=True)
    parser.add_argument("--result-parent-dir", type=str, required=True)
    parser.add_argument("--maxk", type=int, default=30)
    parser.add_argument(
        "--plot-title", type=str, default="Neighborhood Overlap Heatmap"
    )
    parser.add_argument("--downsample-size", type=int, default=None)
    args = parser.parse_args()

    # Set random seed for reproducibility
    seed_all(42)

    # Set up directories for results
    base_dir = os.path.expanduser(args.result_parent_dir)
    plot_dir = os.path.join(
        base_dir, "plots/neighborhood_overlaps/layers_representations"
    )
    results_dir = os.path.join(
        base_dir, "results/neighborhood_overlaps/layers_representations"
    )

    # Create plot directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Create a suffix for the results filenames based on the paths
    split_path1 = args.residual_stream_path1.split("representations", 1)
    split_path2 = args.residual_stream_path2.split("representations", 1)
    remainder1 = split_path1[1].split(".safetensors")[0].split("/")[1:]
    remainder2 = split_path2[1].split(".safetensors")[0].split("/")[1:]
    model_name1 = remainder1[0]
    model_name2 = remainder2[0]
    details = (
        "_".join(remainder1[1::])
        if len("_".join(remainder1[1::])) > len("_".join(remainder2[1::]))
        else "_".join(remainder2[1::])
    )
    details = (
        details.replace(model_name1, "").replace(model_name2, "").replace("lm-_", "")
    )
    suffix_filename = f"{model_name1}_vs_{model_name2}_{details}_maxk-{args.maxk}"
    if "downsample" not in suffix_filename and args.downsample_size is not None:
        suffix_filename += f"_analysis-downsample-{args.downsample_size}"

    layer_overlap_file_name = f"layer-overlap_{suffix_filename}.csv"
    csv_path = os.path.join(
        results_dir,
        layer_overlap_file_name,
    )

    if not os.path.exists(csv_path):
        # Construct file paths
        file_paths = {
            model_name1: os.path.expanduser(args.residual_stream_path1),
            model_name2: os.path.expanduser(args.residual_stream_path2),
        }

        # Compute overlap between the models
        layer_overlaps = analyze_layers(
            file_paths[model_name1],
            file_paths[model_name2],
            maxk=args.maxk,
            downsample_size=args.downsample_size,
        )
        # Save results to a csv file
        layer_overlaps_df = pd.DataFrame(
            list(layer_overlaps.items()), columns=["Layer", "Neighborhood Overlap"]
        )

        layer_overlaps_df.to_csv(csv_path, index=False)
        print(f"Layer-wise neighborhood overlaps saved to {csv_path}")
    else:
        print(f"Layer-wise neighborhood overlaps already computed: {csv_path}")
        layer_overlaps_df = pd.read_csv(csv_path)

    # Generate plot if we have multiple layers
    if len(layer_overlaps_df) != 1:
        plot_path = os.path.join(
            plot_dir,
            f"layer-overlap_{suffix_filename}.png",
        )
        plot_layer_overlaps(layer_overlaps_df, plot_path, plot_title=args.plot_title)


if __name__ == "__main__":
    main()
