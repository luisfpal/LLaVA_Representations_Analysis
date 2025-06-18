import torch
import numpy as np
from typing import Union, Dict, Any, Tuple
from dadapy.data import Data
from anatome.similarity import (
    svcca_distance,
    linear_cka_distance,
)


def _ensure_device(
    *tensors: torch.Tensor,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Move tensors to GPU if available, otherwise keep on CPU.

    Args:
        *tensors: Variable number of torch tensors

    Returns:
        Single tensor if one input, tuple of tensors if multiple inputs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moved_tensors = tuple(tensor.to(device) for tensor in tensors)

    if len(moved_tensors) == 1:
        return moved_tensors[0]
    return moved_tensors


def _preprocess_data(
    data1: Union[np.ndarray, torch.Tensor],
    data2: Union[np.ndarray, torch.Tensor],
    output_format: str = "tensor",
    remove_duplicates: bool = True,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Common preprocessing for similarity computation.

    Args:
        data1: First dataset
        data2: Second dataset
        output_format: 'tensor' or 'numpy'
        remove_duplicates: Whether to remove duplicates and match indices

    Returns:
        Tuple of preprocessed data1 and data2
    """
    # Ensure both datasets have the same number of samples
    if data1.shape[0] != data2.shape[0]:
        raise ValueError(
            f"Data1 has {data1.shape[0]} samples, Data2 has {data2.shape[0]} samples"
        )

    # Convert to numpy arrays for duplicate removal if needed
    if isinstance(data1, torch.Tensor):
        np_data1 = data1.cpu().numpy()
    else:
        np_data1 = data1

    if isinstance(data2, torch.Tensor):
        np_data2 = data2.cpu().numpy()
    else:
        np_data2 = data2

    # Remove duplicates and match indices if requested
    if remove_duplicates:
        unique_data1, indices1 = np.unique(np_data1, axis=0, return_index=True)
        original_num_samples = np_data1.shape[0]
        unique_num_samples = unique_data1.shape[0]
        if unique_num_samples < original_num_samples:
            print(
                f"Warning: {original_num_samples - unique_num_samples} duplicate samples removed."
            )
        matched_data2 = np_data2[indices1]
    else:
        unique_data1 = np_data1
        matched_data2 = np_data2

    # Convert to requested output format
    if output_format == "tensor":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return (
            torch.tensor(unique_data1, dtype=torch.float32, device=device),
            torch.tensor(matched_data2, dtype=torch.float32, device=device),
        )
    else:
        return unique_data1, matched_data2


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
        data1: First dataset
        data2: Second dataset
        maxk: Maximum number of nearest neighbors to consider
        use_dadapy: Whether to use dadapy for distance computation
        metric: Distance metric to use ('euclidean' or 'cosine')

    Returns:
        Overlap score between the two datasets

    Raises:
        ValueError: If the number of samples in data1 and data2 do not match.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_dadapy:
        # For dadapy, we need numpy arrays and should remove duplicates
        processed_data1, processed_data2 = _preprocess_data(
            data1, data2, output_format="numpy", remove_duplicates=True
        )
        d1 = Data(processed_data1)
        d1.compute_distances(maxk=maxk)
        return d1.return_data_overlap(processed_data2)

    # For manual computation, preprocess as tensors with duplicate removal
    data1, data2 = _preprocess_data(
        data1, data2, output_format="tensor", remove_duplicates=True
    )

    with torch.no_grad():
        X_tensor, Y_tensor = _ensure_device(data1, data2)
        X_tensor = X_tensor.to(torch.float32)
        Y_tensor = Y_tensor.to(torch.float32)

        if metric == "cosine":
            X_tensor = torch.nn.functional.normalize(X_tensor, dim=1)
            Y_tensor = torch.nn.functional.normalize(Y_tensor, dim=1)
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
            return (indicator_X * indicator_Y).sum().item() / (maxk * N)

        x_neighbors = x_neighbors.cpu()
        y_neighbors = y_neighbors.cpu()
        overlap_sum = 0.0
        for i in range(N):
            xi = set(x_neighbors[i].tolist())
            yi = set(y_neighbors[i].tolist())
            overlap_sum += len(xi & yi)

        return overlap_sum / (maxk * N)


def compute_distance_correlation(
    latent: Union[np.ndarray, torch.Tensor], control: Union[np.ndarray, torch.Tensor]
):
    """
    Compute distance correlation between two datasets.

    Args:
        latent: First dataset/representation
        control: Second dataset/representation

    Returns:
        Distance correlation score between the two datasets
    """
    # Preprocess and move to GPU if available
    latent, control = _preprocess_data(
        latent, control, output_format="tensor", remove_duplicates=False
    )

    # Ensure tensors are on GPU if available
    latent, control = _ensure_device(latent, control)

    def normalize(data):
        if len(data.shape) > 1 and data.shape[1] > 1:
            return torch.nn.functional.normalize(data, dim=1)
        else:
            return data

    with torch.no_grad():
        latent = normalize(latent)
        control = normalize(control)
        matrix_a = torch.cdist(latent, latent)
        matrix_b = torch.cdist(control, control)
        matrix_A = (
            matrix_a
            - torch.mean(matrix_a, dim=0, keepdim=True)
            - torch.mean(matrix_a, dim=1, keepdim=True)
            + torch.mean(matrix_a)
        )
        matrix_B = (
            matrix_b
            - torch.mean(matrix_b, dim=0, keepdim=True)
            - torch.mean(matrix_b, dim=1, keepdim=True)
            + torch.mean(matrix_b)
        )

        Gamma_XY = torch.sum(matrix_A * matrix_B) / (
            matrix_A.shape[0] * matrix_A.shape[1]
        )
        Gamma_XX = torch.sum(matrix_A * matrix_A) / (
            matrix_A.shape[0] * matrix_A.shape[1]
        )
        Gamma_YY = torch.sum(matrix_B * matrix_B) / (
            matrix_A.shape[0] * matrix_A.shape[1]
        )

        correlation_r = Gamma_XY / torch.sqrt(Gamma_XX * Gamma_YY + 1e-9)
        return correlation_r.item()


def compute_svcca_similarity(
    X: Union[np.ndarray, torch.Tensor], Y: Union[np.ndarray, torch.Tensor]
):
    """
    Compute SVCCA similarity between two datasets.

    Args:
        X: First dataset/representation
        Y: Second dataset/representation

    Returns:
        SVCCA similarity score between the two datasets
    """
    # Preprocess and move to GPU if available
    X, Y = _preprocess_data(X, Y, output_format="tensor", remove_duplicates=False)

    # Ensure tensors are on GPU if available
    X, Y = _ensure_device(X, Y)

    return 1 - svcca_distance(X, Y, accept_rate=0.99, backend="svd")


def compute_linear_cka_similarity(
    X: Union[np.ndarray, torch.Tensor], Y: Union[np.ndarray, torch.Tensor]
):
    """
    Compute Linear CKA similarity between two datasets.

    Args:
        X: First dataset/representation
        Y: Second dataset/representation

    Returns:
        Linear CKA similarity score between the two datasets
    """
    # Preprocess and move to GPU if available
    X, Y = _preprocess_data(X, Y, output_format="tensor", remove_duplicates=False)

    # Ensure tensors are on GPU if available
    X, Y = _ensure_device(X, Y)

    return 1 - linear_cka_distance(X, Y, reduce_bias=False)


def compute_similarity(
    data1: Union[np.ndarray, torch.Tensor],
    data2: Union[np.ndarray, torch.Tensor],
    measure: str,
    **kwargs,
) -> float:
    """
    Unified function to compute similarity between two neural network representations.

    Args:
        data1: First dataset/representation
        data2: Second dataset/representation
        measure: Similarity measure name. Options:
            - 'neighborhood_overlap': Neighborhood overlap measure
            - 'distance_correlation': Distance correlation measure
            - 'svcca': SVCCA similarity measure
            - 'linear_cka': Linear CKA similarity measure
        **kwargs: Additional parameters specific to each measure:
            For 'neighborhood_overlap':
                - maxk (int): Maximum number of nearest neighbors (default: 30)
                - use_dadapy (bool): Whether to use dadapy (default: False)
                - metric (str): Distance metric 'euclidean' or 'cosine' (default: 'euclidean')

    Returns:
        Similarity score between the two representations

    Raises:
        ValueError: If measure name is not recognized or data shapes don't match
    """
    available_measures = {
        "neighborhood_overlap",
        "distance_correlation",
        "svcca",
        "linear_cka",
    }

    if measure not in available_measures:
        raise ValueError(
            f"Unknown measure '{measure}'. Available measures: {available_measures}"
        )

    if measure == "neighborhood_overlap":
        maxk = kwargs.get("maxk", 30)
        use_dadapy = kwargs.get("use_dadapy", False)
        metric = kwargs.get("metric", "euclidean")
        return compute_neighborhood_overlap(data1, data2, maxk, use_dadapy, metric)

    elif measure == "distance_correlation":
        return compute_distance_correlation(data1, data2)

    elif measure == "svcca":
        return compute_svcca_similarity(data1, data2)

    elif measure == "linear_cka":
        return compute_linear_cka_similarity(data1, data2)


# Backward compatibility aliases
def distance_correlation(latent, control):
    """Backward compatibility alias for compute_distance_correlation."""
    return compute_distance_correlation(latent, control)


def svcca_similarity(X, Y):
    """Backward compatibility alias for compute_svcca_similarity."""
    return compute_svcca_similarity(X, Y)


def linear_cka_similarity(X, Y):
    """Backward compatibility alias for compute_linear_cka_similarity."""
    return compute_linear_cka_similarity(X, Y)
