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
    compute_similarity,
    seed_all,
    sample_unique_row_indices,
    create_filename_from_paths,
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


def compute_matrix_similarities(
    network1_layers_heads_representations: Dict[str, torch.Tensor],
    network2_layers_heads_representations: Dict[str, torch.Tensor],
    measure: str = "neighborhood_overlap",
    maxk: Optional[int] = 30,
    downsample_size: Optional[int] = None,
    accept_rate: Optional[float] = None,
) -> pd.DataFrame:
    """
    Computes the similarity measure matrix between two networks' layer-head representations.

    Args:
        network1_layers_heads_representations (Dict[str, torch.Tensor]): Representations from network 1.
        network2_layers_heads_representations (Dict[str, torch.Tensor]): Representations from network 2.
        measure (str): The similarity measure to compute.
        maxk (Optional[int]): The 'k' in top-k neighbors to consider.
        downsample_size (Optional[int]): The number of samples to downsample to.
        accept_rate (Optional[float]): The accept rate for the SVCCA similarity measure.
    Returns:
        pd.DataFrame: A DataFrame containing the similarity measure matrix indexed by layers and heads.
    """
    layers1, heads1 = count_layers_and_heads(network1_layers_heads_representations)
    layers2, heads2 = count_layers_and_heads(network2_layers_heads_representations)
    if (layers1 != layers2) or (heads1 != heads2):
        raise ValueError(
            "The number of layers or heads in the two networks do not match."
        )
    num_layers1 = len(layers1)
    num_heads1 = len(heads1)
    matrix_similarities = np.zeros((num_layers1, num_heads1), dtype=np.float32)

    reduced_unique_sample_indices = None
    data_size = None
    with tqdm(
        total=num_layers1, desc=f"Computing {measure} Matrix", unit="layer"
    ) as pbar:
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

                    similarity_measure = compute_similarity(
                        data1=R1,
                        data2=R2,
                        measure=measure,
                        maxk=maxk,
                        accept_rate=accept_rate,
                    )
                    matrix_similarities[layer_idx, head_idx] = similarity_measure
            pbar.update(1)

    matrix_similarities_df = pd.DataFrame(
        matrix_similarities,
        index=[f"{layer + 1}" for layer in layers1],
        columns=[f"{head + 1}" for head in heads1],
    )

    return matrix_similarities_df


def plot_similarity_measure_heatmap(
    matrix_similarities_df: pd.DataFrame,
    title: str = "Neighborhood Overlap Heatmap",
    xlabel: str = "Heads",
    ylabel: str = "Layers",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 11),
    dpi: int = 200,
    percentage_threshold: Optional[float] = None,
    top_p_percent: Optional[float] = None,
    bottom_p_percent: Optional[float] = None,
    measure: str = "neighborhood_overlap",
) -> None:
    if not isinstance(matrix_similarities_df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame for 'matrix_similarities_df'.")

    for name, val in [
        ("percentage_threshold", percentage_threshold),
        ("top_p_percent", top_p_percent),
        ("bottom_p_percent", bottom_p_percent),
    ]:
        if val is not None and not (0 <= val <= 1):
            raise ValueError(f"{name} must be between 0 and 1")

    plt.figure(figsize=figsize)

    # ---------------------- BASE METRICS ----------------------
    max_val_global = matrix_similarities_df.max().max()
    min_val_global = matrix_similarities_df.min().min()
    print(f"Max {measure}: {max_val_global:.2f}, Min {measure}: {min_val_global:.2f}")

    # ------------------ THRESHOLD BOUNDARIES ------------------
    if percentage_threshold is not None:
        max_layer = matrix_similarities_df.max(axis=1)
        min_layer = matrix_similarities_df.min(axis=1)
        range_layer = max_layer - min_layer

        upper_bound = max_layer - (range_layer * percentage_threshold)
        lower_bound = min_layer + (range_layer * percentage_threshold)

        threshold_mask = (matrix_similarities_df >= upper_bound.values[:, None]) | (
            matrix_similarities_df <= lower_bound.values[:, None]
        )

        min_upper_bound = upper_bound.min()
        max_lower_bound = lower_bound.max()
        print(
            f"Min upper bound: {min_upper_bound:.4f}, Max lower bound: {max_lower_bound:.4f}"
        )

        # Percentage of values in the upper and lower bounds
        num_values_in_upper_bound = (
            (matrix_similarities_df >= upper_bound.values[:, None]).sum().sum()
        )
        num_values_in_lower_bound = (
            (matrix_similarities_df <= lower_bound.values[:, None]).sum().sum()
        )
        percentage_in_upper_bound = (
            num_values_in_upper_bound / matrix_similarities_df.size
        )
        percentage_in_lower_bound = (
            num_values_in_lower_bound / matrix_similarities_df.size
        )
        print(
            f"% in upper bound (white): {percentage_in_upper_bound * 100:.2f}%, # values: {num_values_in_upper_bound}"
        )
        print(
            f"% in lower bound (white): {percentage_in_lower_bound * 100:.2f}%, # values: {num_values_in_lower_bound}"
        )
        total_values_in_bounds = num_values_in_upper_bound + num_values_in_lower_bound
        print(
            f"% in bounds (white): {total_values_in_bounds / matrix_similarities_df.size * 100:.2f}%, # values: {total_values_in_bounds}"
        )
    else:
        threshold_mask = pd.DataFrame(
            False,
            index=matrix_similarities_df.index,
            columns=matrix_similarities_df.columns,
        )  # all are valid

    # Get valid (non-NaN) values for percentile calculations
    valid_values = matrix_similarities_df.values.flatten()
    valid_values = valid_values[~np.isnan(valid_values)]
    total_values = matrix_similarities_df.size
    valid_count = len(valid_values)

    # ---------- TOP p% ----------
    top_p_mask = pd.DataFrame(
        False,
        index=matrix_similarities_df.index,
        columns=matrix_similarities_df.columns,
    )
    if top_p_percent is not None and top_p_percent > 0 and valid_count > 0:
        # Use percentile instead of partition for more robust calculation
        top_cutoff = np.percentile(valid_values, (1 - top_p_percent) * 100)
        top_p_mask = matrix_similarities_df >= top_cutoff
        actual_top_percentage = top_p_mask.sum().sum() / total_values
        print(f"Top {top_p_percent * 100:.2f}% cutoff value: {top_cutoff:.4f}")
        print(
            f"% identified as top p%: {actual_top_percentage * 100:.2f}%, # values: {top_p_mask.sum().sum()}"
        )

    # ---------- BOTTOM p% ----------
    bottom_p_mask = pd.DataFrame(
        False,
        index=matrix_similarities_df.index,
        columns=matrix_similarities_df.columns,
    )
    if bottom_p_percent is not None and bottom_p_percent > 0 and valid_count > 0:
        # Use percentile instead of partition for more robust calculation
        bottom_cutoff = np.percentile(valid_values, bottom_p_percent * 100)
        bottom_p_mask = matrix_similarities_df <= bottom_cutoff
        actual_bottom_percentage = bottom_p_mask.sum().sum() / total_values
        print(f"Bottom {bottom_p_percent * 100:.2f}% cutoff value: {bottom_cutoff:.4f}")
        print(
            f"% identified as bottom p%: {actual_bottom_percentage * 100:.2f}%, # values: {bottom_p_mask.sum().sum()}"
        )

    # ------------------ FINAL MASKS ------------------
    final_annotation_mask = threshold_mask  # Only annotate if outside threshold
    final_red_mask = (
        top_p_mask & threshold_mask
    )  # Only color red if top and outside threshold
    final_orange_mask = (
        bottom_p_mask & threshold_mask
    )  # Only color orange if bottom and outside threshold

    # Final percentage of values in the final masks
    total_values_in_red_mask = final_red_mask.sum().sum()
    total_values_in_orange_mask = final_orange_mask.sum().sum()
    final_red_percentage = total_values_in_red_mask / total_values
    final_orange_percentage = total_values_in_orange_mask / total_values
    total_values_in_red_and_orange_mask = (
        total_values_in_red_mask + total_values_in_orange_mask
    )
    print(
        f"% in top p% (red): {final_red_percentage * 100:.2f}%, # values: {total_values_in_red_mask}"
    )
    print(
        f"% in bottom p% (orange): {final_orange_percentage * 100:.2f}%, # values: {total_values_in_orange_mask}"
    )
    print(
        f"% in top and bottom p% (red and orange): {total_values_in_red_and_orange_mask / total_values * 100:.2f}%, # values: {total_values_in_red_and_orange_mask}"
    )

    # ------------------ ANNOTATION LABELS ------------------
    annot_labels = matrix_similarities_df.where(final_annotation_mask).map(
        lambda x: f"{x:.2f}"
    )
    annot_labels = annot_labels.replace("nan", "")

    # ------------------ HEATMAP
    heatmap_kwargs = {
        "data": matrix_similarities_df,
        "fmt": "",  # we already formatted the numbers in annot_labels
        "cmap": "viridis",
        "annot_kws": {"size": 8, "weight": "bold"},  # General styling for annotations
        "annot": annot_labels,
    }

    if min_val_global > 0.0:
        heatmap_kwargs["vmax"] = 1
    ax = sns.heatmap(**heatmap_kwargs)

    # Titles and labels
    ax.set_title(f"{title} [{measure}]", fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticklabels(matrix_similarities_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix_similarities_df.index, rotation=0)
    ax.tick_params(axis="both", labelsize=12)

    # ------------------ MANUAL CELL ANNOTATION ------------------
    for i in range(matrix_similarities_df.shape[0]):
        for j in range(matrix_similarities_df.shape[1]):
            label = annot_labels.iat[i, j]
            if label:
                color = (
                    "red"
                    if final_red_mask.iat[i, j]
                    else "darkorange"
                    if final_orange_mask.iat[i, j]
                    else "white"
                )
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    label,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                    weight="bold",
                )

    # Add background color and style
    plt.gca().set_facecolor("#F8F8F8")
    plt.gcf().set_facecolor("white")

    plt.tight_layout()
    if save_path:
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
        description="Analyze similarity measures between LLMs heads representations extraction in residual stream from two models"
    )
    parser.add_argument("--heads-residual-stream-path1", type=str, required=True)
    parser.add_argument("--heads-residual-stream-path2", type=str, required=True)
    parser.add_argument("--result-parent-dir", type=str, required=True)
    parser.add_argument("--maxk", type=int, default=30)
    parser.add_argument("--plot-title", type=str, default="Similarity Measure Heatmap")
    parser.add_argument("--downsample-size", type=int, default=None)
    parser.add_argument("--measure", type=str, default="neighborhood_overlap")
    parser.add_argument("--accept-rate", type=float, default=0.95)
    args = parser.parse_args()

    # Set random seed for reproducibility
    seed_all(42)

    # Set up directories for results
    base_dir = os.path.expanduser(args.result_parent_dir)
    plot_dir = os.path.join(base_dir, f"plots/{args.measure}/heads_representations")
    results_dir = os.path.join(
        base_dir, f"results/{args.measure}/heads_representations"
    )

    # Create plot directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Create a suffix for the results filenames based on the paths
    filename, model_name1, model_name2 = create_filename_from_paths(
        args.heads_residual_stream_path1,
        args.heads_residual_stream_path2,
        args,
    )

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
        # Compute similarity measure between the models
        start_time = time.time()
        matrix_similarities_df = compute_matrix_similarities(
            network1_representations,
            network2_representations,
            maxk=args.maxk,
            downsample_size=args.downsample_size,
            measure=args.measure,
            accept_rate=args.accept_rate,
        )
        elapsed_time = time.time() - start_time
        print(
            f"✅ {args.measure} matrix between {model_name1} and {model_name2} computed successfully."
        )
        print(f"⏱️ Time taken: {elapsed_time / 60:.2f} minutes")

        matrix_similarities_df.to_parquet(data_path_parquet)
        matrix_similarities_df.to_csv(data_path_csv)
        print(f"{args.measure} matrix saved to {data_path_parquet}")

    else:
        # If the data file already exists, load the results from it
        matrix_similarities_df = pd.read_parquet(data_path_parquet)
        print(f"Loaded {args.measure} matrix from {data_path_parquet}")
    # Generate the heatmap plot
    plot_path = os.path.join(
        plot_dir,
        f"{filename}.png",
    )

    plot_similarity_measure_heatmap(
        matrix_similarities_df=matrix_similarities_df,
        title=args.plot_title,
        xlabel="Heads",
        ylabel="Layers",
        save_path=plot_path,
        percentage_threshold=0.1,
        top_p_percent=0.05,
        bottom_p_percent=0.03,
        measure=args.measure,
    )


if __name__ == "__main__":
    main()
