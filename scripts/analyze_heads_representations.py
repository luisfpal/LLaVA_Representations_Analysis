import os
import torch
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
    plot_similarity_measure_heatmap,
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
