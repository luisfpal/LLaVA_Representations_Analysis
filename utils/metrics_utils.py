import torch
import numpy as np
from typing import Union
from dadapy.data import Data


def compute_neighborhood_overlap(
    data1: Union[np.ndarray, torch.Tensor],
    data2: Union[np.ndarray, torch.Tensor],
    maxk: int = 30,
    use_dadapy: bool = False,
    metric: str = "euclidean",
):
    """
    Compute neighborhood overlap between two datasets.

    Args:
        data1: First dataset (numpy array)
        data2: Second dataset (numpy array)
        maxk: Maximum number of nearest neighbors to consider
        use_dadapy: Whether to use dadapy for distance computation
        metric: Distance metric to use ('euclidean' or 'cosine')
    Raises:
        ValueError: If the number of samples in data1 and data2 do not match.

    Returns:
        Overlap score between the two datasets
    """
    # Ensure both datasets have the same number of samples
    if data1.shape[0] != data2.shape[0]:
        raise ValueError(
            f"Data1 has {data1.shape[0]} samples, Data2 has {data2.shape[0]} samples"
        )

    # Convert to numpy arrays if needed
    if isinstance(data1, torch.Tensor):
        data1 = data1.cpu().numpy()
    if isinstance(data2, torch.Tensor):
        data2 = data2.cpu().numpy()

    # Remove duplicates and match indices
    unique_X, indices1 = np.unique(data1, axis=0, return_index=True)
    original_num_samples = data1.shape[0]
    unique_num_samples = unique_X.shape[0]
    if unique_num_samples < original_num_samples:
        print(
            f"Warning: {original_num_samples - unique_num_samples} duplicate samples removed."
        )
    matched_Y = data2[indices1]

    if use_dadapy:
        d1 = Data(unique_X)
        d1.compute_distances(maxk=maxk)
        return d1.return_data_overlap(matched_Y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        X_tensor = torch.tensor(unique_X, dtype=torch.float32, device=device)
        Y_tensor = torch.tensor(matched_Y, dtype=torch.float32, device=device)

        if metric == "cosine":
            X_tensor = torch.nn.functional.normalize(X_tensor)
            Y_tensor = torch.nn.functional.normalize(Y_tensor)
            x_neighbors = torch.topk(X_tensor @ X_tensor.T, maxk + 1, dim=1).indices[
                :, 1:
            ]
            y_neighbors = torch.topk(Y_tensor @ Y_tensor.T, maxk + 1, dim=1).indices[
                :, 1:
            ]
        else:
            x_neighbors = torch.topk(
                torch.cdist(X_tensor, X_tensor), maxk + 1, dim=1, largest=False
            ).indices[:, 1:]
            y_neighbors = torch.topk(
                torch.cdist(Y_tensor, Y_tensor), maxk + 1, dim=1, largest=False
            ).indices[:, 1:]

        N = x_neighbors.shape[0]
        if device.type == "cuda":
            indicator_X = torch.zeros((N, N), dtype=torch.int16, device=device)
            indicator_Y = torch.zeros((N, N), dtype=torch.int16, device=device)
            indicator_X.scatter_(dim=1, index=x_neighbors, value=1)
            indicator_Y.scatter_(dim=1, index=y_neighbors, value=1)
            # broadcasting for value
            return (indicator_X * indicator_Y).sum().item() / (maxk * N)
            # Hadamard product for overlap

        x_neighbors = x_neighbors.cpu()
        y_neighbors = y_neighbors.cpu()
        # Compute neighborhood overlap via set intersection (per row)
        overlap_sum = 0.0
        for i in range(N):
            xi = set(x_neighbors[i].tolist())
            yi = set(y_neighbors[i].tolist())
            overlap_sum += len(xi & yi)

        return overlap_sum / (maxk * N)
