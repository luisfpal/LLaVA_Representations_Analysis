import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import argparse
from safetensors.torch import load_file
import time
from tqdm import tqdm
from utils import (
    compute_neighborhood_overlap,
    seed_all,
    sample_unique_row_indices,
    create_filename_suffix_from_paths,
)


def count_layers_and_heads(tensor_dict: Dict[str, torch.Tensor]) -> Tuple[List, List]:
    """
    Extract unique layers and heads from dictionary keys.

    Args:
        tensor_dict (Dict[str, torch.Tensor]): Keys formatted as 'layer_<num>/head_<num>'

    Returns:
        Tuple[list, list]: lists of layer indices and head indices
    """
    layers = set()
    heads = set()

    for key in tensor_dict:
        try:
            layer_part, head_part = key.split("/")
            layer_idx = int(layer_part.split("_")[1])
            head_idx = int(head_part.split("_")[1])
            layers.add(layer_idx)
            heads.add(head_idx)
        except (IndexError, ValueError):
            raise ValueError(
                f"Invalid key format: {key}. Expected 'layer_<num>/head_<num>'"
            )

    return list(layers), list(heads)


def compute_neighborhood_overlap_matrix(
    network1_layers_heads_representations: Dict[str, torch.Tensor],
    network2_layers_heads_representations: Dict[str, torch.Tensor],
    maxk: int = 10,
    downsample_size: Optional[int] = None,
) -> Tuple[pd.DataFrame]:
    """
    Computes the neighborhood overlap matrix between two networks' layer-head representations.

    Args:
        network1_layers_heads_representations (Dict[str, torch.Tensor]): Representations from network 1.
        network2_layers_heads_representations (Dict[str, torch.Tensor]): Representations from network 2.
        maxk (int): The 'k' in top-k neighbors to consider.

    Returns:
        Tuple[pd.DataFrame]: A DataFrame containing the overlap matrix indexed by layers and heads.
    """
    layers1, heads1 = count_layers_and_heads(network1_layers_heads_representations)
    layers2, heads2 = count_layers_and_heads(network2_layers_heads_representations)
    if (layers1 != layers2) or (heads1 != heads2):
        raise ValueError(
            "The number of layers or heads in the two networks do not match."
        )
    num_layers1 = len(layers1)
    num_heads1 = len(heads1)
    overlap_matrix = np.zeros((num_layers1, num_heads1), dtype=np.float32)

    total = num_layers1 * num_heads1
    reduced_unique_sample_indices = None
    data_size = None
    with tqdm(total=total, desc="Computing Overlap Matrix") as pbar:
        for layer_idx, layer_key in enumerate(layers1):
            for head_idx, head_key in enumerate(heads1):
                key = f"layer_{layer_key}/head_{head_key}"
                if (
                    key in network1_layers_heads_representations
                    and key in network2_layers_heads_representations
                ):
                    R1 = network1_layers_heads_representations[key]
                    R2 = network2_layers_heads_representations[key]
                    if layer_idx == 0 and head_idx == 0:
                        data_size = R1.size(0)
                        if downsample_size is not None and data_size > downsample_size:
                            # Sample unique row indices only once for the first layer-head pair
                            reduced_unique_sample_indices = sample_unique_row_indices(
                                R1, downsample_size
                            )
                            print(
                                f"Downsampling from {data_size} to {downsample_size} unique instances."
                            )

                    if reduced_unique_sample_indices is not None:
                        R1 = R1[reduced_unique_sample_indices]
                        R2 = R2[reduced_unique_sample_indices]

                    overlap = compute_neighborhood_overlap(
                        data1=R1,
                        data2=R2,
                        maxk=maxk,
                    )
                    overlap_matrix[layer_idx, head_idx] = overlap
                pbar.update(1)

    overlap_matrix_df = pd.DataFrame(
        overlap_matrix,
        index=[f"{layer + 1}" for layer in layers1],
        columns=[f"{head + 1}" for head in heads1],
    )

    return overlap_matrix_df


def plot_overlap_heatmap(
    overlap_df: pd.DataFrame,
    title: str = "Neighborhood Overlap Heatmap",
    xlabel: str = "Heads",
    ylabel: str = "Layers",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 11),
    dpi: int = 200,
    min_thresh: Optional[float] = None,  # Annotate values <= this
    max_thresh: Optional[float] = None,  # Annotate values >= this
    percentage_thresh: Optional[float] = None,
):
    """
    Plots a heatmap of the neighborhood overlap matrix using a DataFrame.

    Args:
        overlap_df (pd.DataFrame): DataFrame with overlap values, indexed by layers and labeled by heads.
        title (str): Title for the heatmap.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        save_path (str): File path to save the plot.
        min_thresh (float, optional): If set, annotates values less than or
                                       equal to this threshold.
        max_thresh (float, optional): If set, annotates values greater than or
                                        equal to this threshold.
    """
    if not isinstance(overlap_df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame for 'overlap_df'.")

    plt.figure(figsize=figsize)  # (W, H)

    # Initialize an empty DataFrame for annotation text.
    # We will fill this with formatted numbers where conditions are met.
    annot_labels = pd.DataFrame("", index=overlap_df.index, columns=overlap_df.columns)

    # Initialize a mask for all annotations. It's False everywhere by default.
    final_mask = pd.DataFrame(False, index=overlap_df.index, columns=overlap_df.columns)

    final_mask = overlap_df < 0.0  # Initialize mask for negative values

    max_value_global = overlap_df.max().max()
    print(f"Max overall overlap value: {max_value_global:.2f}")
    min_value_global = overlap_df.min().min()
    print(f"Min overall overlap value: {min_value_global:.2f}")

    # Update the mask if thresholds are provided
    if max_thresh is not None:
        final_mask = final_mask | (overlap_df >= max_thresh)
    if min_thresh is not None:
        final_mask = final_mask | (overlap_df <= min_thresh)
    if percentage_thresh is not None:
        max_value_per_layer = overlap_df.max(axis=1)
        min_value_per_layer = overlap_df.min(axis=1)

        range_per_layer = max_value_per_layer - min_value_per_layer
        threshold_amount_per_layer = range_per_layer * percentage_thresh

        upper_bound_for_annotation = max_value_per_layer - threshold_amount_per_layer
        lower_bound_for_annotation = min_value_per_layer + threshold_amount_per_layer

        # Condition for annotating: value is in the top/bottom percentage of its layer's range
        condition_met = (overlap_df >= upper_bound_for_annotation.values[:, None]) | (
            overlap_df <= lower_bound_for_annotation.values[:, None]
        )

        final_mask = final_mask | condition_met

        max_bound_overall = upper_bound_for_annotation.max()
        min_bound_overall = lower_bound_for_annotation.min()
        print(f"Max overall bound for annotation: {max_bound_overall:.2f}")
        print(f"Min overall bound for annotation: {min_bound_overall:.2f}")

    # Apply the mask to create the labels.
    # Where the mask is True, format the number; otherwise, it remains an empty string.
    annot_labels = overlap_df[final_mask].map(
        lambda x: f"{x:.2f}"
    )  # Changed applymap to map
    annot_labels = annot_labels.replace(
        "nan", ""
    )  # Replace string "nan" with empty string

    heatmap_kwargs = {
        "data": overlap_df,
        "fmt": "",  # we already formatted the numbers in annot_labels
        "cmap": "viridis",
        "annot_kws": {"size": 8, "weight": "bold"},  # General styling for annotations
        "annot": annot_labels,
    }

    if min_value_global > 0.0:
        heatmap_kwargs["vmax"] = 1

    ax = sns.heatmap(
        **heatmap_kwargs,
    )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    # Optional: rotate x and y labels if needed
    ax.set_xticklabels(overlap_df.columns, rotation=45, ha="right")  # heads
    ax.set_yticklabels(overlap_df.index, rotation=0)  # layers
    ax.tick_params(axis="both", which="major", labelsize=12)

    # Add background color and style
    plt.gca().set_facecolor("#F8F8F8")
    plt.gcf().set_facecolor("white")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()


def load_heads_representations(
    file_path: str,
) -> Dict[str, torch.Tensor]:
    """
    Load head representations from a file.

    Args:
        file_path (str): Path to the file containing head representations.

    Returns:
        Dict[str, torch.Tensor]: Dictionary of head representations.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    tensors_dict = load_file(file_path)

    if not isinstance(tensors_dict, dict):
        raise TypeError("Expected a dictionary of tensors.")

    return tensors_dict


def main():
    parser = argparse.ArgumentParser(
        description="Analyze neighborhood overlap between LLMs heads representations extraction in residual stream from two models"
    )
    parser.add_argument("--heads-residual-stream-path1", type=str, required=True)
    parser.add_argument("--heads-residual-stream-path2", type=str, required=True)
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
        base_dir, "plots/neighborhood_overlaps/heads_representations"
    )
    results_dir = os.path.join(
        base_dir, "results/neighborhood_overlaps/heads_representations"
    )

    # Create plot directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Create a suffix for the results filenames based on the paths
    suffix_filename, model_name1, model_name2 = create_filename_suffix_from_paths(
        args.heads_residual_stream_path1,
        args.heads_residual_stream_path2,
        args,
    )
    filename = f"layer-head-overlap_{suffix_filename}"
    data_path_parquet = os.path.join(results_dir, f"{filename}.parquet")
    data_path_csv = os.path.join(results_dir, f"{filename}.csv")

    if not os.path.exists(data_path_parquet):
        # Load the head representations from the specified paths
        file_paths = {
            model_name1: os.path.expanduser(args.heads_residual_stream_path1),
            model_name2: os.path.expanduser(args.heads_residual_stream_path2),
        }
        print(
            f"Loading representations from: \n{file_paths[model_name1]}\n and \n{file_paths[model_name2]}"
        )
        start_time = time.time()
        network1_representations = load_heads_representations(file_paths[model_name1])
        network2_representations = load_heads_representations(file_paths[model_name2])
        elapsed_time = time.time() - start_time
        print(
            f"✅ Representations loaded successfully in {elapsed_time / 60:.2f} minutes."
        )
        # Compute overlap between the models
        start_time = time.time()
        overlap_matrix_df = compute_neighborhood_overlap_matrix(
            network1_representations,
            network2_representations,
            maxk=args.maxk,
            downsample_size=args.downsample_size,
        )
        elapsed_time = time.time() - start_time
        print(
            f"✅ Neighborhood overlap matrix between {model_name1} and {model_name2} computed successfully."
        )
        print(f"⏱️ Time taken: {elapsed_time / 60:.2f} minutes")

        overlap_matrix_df.to_parquet(data_path_parquet)
        overlap_matrix_df.to_csv(data_path_csv)
        print(f"Neighborhood overlap matrix saved to {data_path_parquet}")

    else:
        # If the data file already exists, load the results from it
        overlap_matrix_df = pd.read_parquet(data_path_parquet)
        print(f"Loaded neighborhood overlap matrix from {data_path_parquet}")
    # Generate the heatmap plot
    plot_path = os.path.join(
        plot_dir,
        f"overlap_heatmap_{suffix_filename}.png",
    )

    plot_overlap_heatmap(
        overlap_df=overlap_matrix_df,
        title=args.plot_title,
        xlabel="Heads",
        ylabel="Layers",
        save_path=plot_path,
        percentage_thresh=0.2,
    )


if __name__ == "__main__":
    main()
