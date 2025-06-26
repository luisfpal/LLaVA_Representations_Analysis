import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Optional
from utils import (
    seed_all,
    sample_unique_row_indices,
    create_filename_from_paths,
    compute_similarity,
    load_layers_residual_stream,
)


def plot_layer_similarity(
    similarity_measures_df: pd.DataFrame,
    save_path: str,
    plot_title: str = "Layer-wise Similarity Measure",
    measure: str = "neighborhood_overlap",
) -> None:
    """
    Create a plot of layer-wise similarity measures from a DataFrame.

    Args:
        similarity_measures_df: DataFrame with columns ["layer_index", "similarity_measure"]
        save_path: Path to save the plot
        plot_title: Title for the plot
        measure: Similarity measure to plot
    """
    # Ensure the data is sorted by the layer index
    df = similarity_measures_df.sort_values(by="layer_index").reset_index(drop=True)

    layers = df["layer_index"].values
    similarity_values = df[measure].values

    # Create the plot with good aesthetics
    plt.figure(figsize=(10, 6))
    plt.plot(
        layers,
        similarity_values,
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
    plt.ylabel(measure, fontsize=14)
    plt.title(plot_title, fontsize=16)

    # Set axis limits with a bit of padding
    plt.ylim([max(0, min(similarity_values) - 0.05), 1.05])

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
    residual_stream_data1_path: str,
    residual_stream_data2_path: str,
    measure: str = "neighborhood_overlap",
    maxk: Optional[int] = None,
    downsample_size: Optional[int] = None,
    sigma: Optional[float] = None,
    accept_rate: Optional[float] = None,
) -> Dict[int, float]:
    """
    Compute a similarity measure between models layer by layer.

    Args:
        residual_stream_data1_path: Path to the first model's residual stream file
        residual_stream_data2_path: Path to the second model's residual stream file
        measure: Similarity measure to compute
        maxk: Maximum number of nearest neighbors to consider
        downsample_size: Maximum number of samples to consider
        sigma: Sigma for the RBF kernel
        accept_rate: Accept rate for the SVCCA similarity measure
    Returns:
        Dictionary mapping layer indices to similarity scores
    """
    # Load data from both models
    residual_stream_data1_path = os.path.expanduser(residual_stream_data1_path)
    residual_stream_data2_path = os.path.expanduser(residual_stream_data2_path)
    is_multi_layer1, data1 = load_layers_residual_stream(residual_stream_data1_path)
    is_multi_layer2, data2 = load_layers_residual_stream(residual_stream_data2_path)

    # Ensure both have the same format (multi-layer or single-layer)
    if is_multi_layer1 != is_multi_layer2:
        raise ValueError(
            "Both models must have the same layer structure (single or multi-layer)"
        )

    similarity_measures = {}
    reduced_unique_sample_indices = None
    data_size = None
    if is_multi_layer1:
        # Multi-layer case
        # Find common layers between the two models
        common_layers = sorted(set(data1.keys()).intersection(set(data2.keys())))

        if not common_layers:
            raise ValueError("No common layers found between the two models")

        print(f"Found {len(common_layers)} common layers: {common_layers}")

        # Compute similarity measure for each common layer
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
            similarity_measure = compute_similarity(
                R1, R2, measure=measure, maxk=maxk, sigma=sigma, accept_rate=accept_rate
            )
            print(f"Layer {layer_idx} {measure}: {similarity_measure:.4f}")
            similarity_measures[layer_idx] = similarity_measure
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
        similarity_measure = compute_similarity(
            R1, R2, measure=measure, maxk=maxk, sigma=sigma, accept_rate=accept_rate
        )
        similarity_measures[-1] = (
            similarity_measure  # Use -1 to indicate the default layer
        )
        print(f"Single layer {measure}: {similarity_measure:.4f}")

    return similarity_measures


def main():
    parser = argparse.ArgumentParser(
        description="Analyze residual stream similarity measures"
    )
    parser.add_argument("--residual-stream-path1", type=str, required=True)
    parser.add_argument("--residual-stream-path2", type=str, required=True)
    parser.add_argument("--result-parent-dir", type=str, required=True)
    parser.add_argument("--maxk", type=int, default=30)
    parser.add_argument("--plot-title", type=str, default="Similarity Measure Plot")
    parser.add_argument("--downsample-size", type=int, default=None)
    parser.add_argument("--measure", type=str, default="neighborhood_overlap")
    parser.add_argument("--accept-rate", type=float, default=0.95)
    parser.add_argument("--sigma", type=float, default=0.1)
    args = parser.parse_args()

    # Set random seed for reproducibility
    seed_all(42)

    # Set up directories for results
    base_dir = os.path.expanduser(args.result_parent_dir)
    plot_dir = os.path.join(base_dir, f"plots/{args.measure}/layers_representations")
    results_dir = os.path.join(
        base_dir, f"results/{args.measure}/layers_representations"
    )

    # Create plot directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Create a suffix for the results filenames based on the paths
    filename, model_name1, model_name2 = create_filename_from_paths(
        args.residual_stream_path1,
        args.residual_stream_path2,
        args,
    )

    csv_path = os.path.join(results_dir, f"{filename}.csv")

    if not os.path.exists(csv_path):
        # Construct file paths
        file_paths = {
            model_name1: os.path.expanduser(args.residual_stream_path1),
            model_name2: os.path.expanduser(args.residual_stream_path2),
        }

        # Compute overlap between the models
        similarity_measures = analyze_layers(
            file_paths[model_name1],
            file_paths[model_name2],
            maxk=args.maxk,
            downsample_size=args.downsample_size,
            measure=args.measure,
            sigma=args.sigma,
            accept_rate=args.accept_rate,
        )
        # Save results to a csv file
        similarity_measures_df = pd.DataFrame(
            list(similarity_measures.items()),
            columns=["layer_index", args.measure],
        )

        similarity_measures_df.to_csv(csv_path, index=False)
        print(f"Layer-wise {args.measure} saved to {csv_path}")
    else:
        print(f"Layer-wise {args.measure} already computed: {csv_path}")
        similarity_measures_df = pd.read_csv(csv_path)

    # Generate plot if we have multiple layers
    if len(similarity_measures_df) != 1:
        plot_path = os.path.join(
            plot_dir,
            f"{filename}.png",
        )
        plot_layer_similarity(
            similarity_measures_df,
            plot_path,
            plot_title=args.plot_title,
            measure=args.measure,
        )


if __name__ == "__main__":
    main()
