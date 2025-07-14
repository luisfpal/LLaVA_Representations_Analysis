import random
import numpy as np
import torch
from typing import Union


def seed_all(seed: int):
    """
    Set the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to set.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sample_unique_row_indices(
    tensor: Union[torch.Tensor, np.ndarray], n: int
) -> Union[torch.Tensor, np.ndarray]:
    """
    Returns indices of `n` unique rows randomly sampled from a 2D tensor or array.

    Args:
        tensor (torch.Tensor or np.ndarray): A 2D input tensor/array of shape (rows, cols).
        n (int): The number of unique row indices to sample.

    Returns:
        torch.Tensor or np.ndarray: A 1D tensor or array of indices with shape (n,)
                                     corresponding to sampled rows.
    """
    # Check input type
    if isinstance(tensor, torch.Tensor):
        if tensor.dim() != 2:
            raise ValueError("Input torch tensor must be 2D.")
        num_rows = tensor.size(0)
        if n > num_rows:
            raise ValueError(
                f"Cannot sample {n} unique rows from a tensor with only {num_rows} rows."
            )
        return torch.randperm(num_rows)[:n]

    elif isinstance(tensor, np.ndarray):
        if tensor.ndim != 2:
            raise ValueError("Input NumPy array must be 2D.")
        num_rows = tensor.shape[0]
        if n > num_rows:
            raise ValueError(
                f"Cannot sample {n} unique rows from an array with only {num_rows} rows."
            )
        return np.random.permutation(num_rows)[:n]
