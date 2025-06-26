import math
import torch
import numpy as np
from typing import Union, Tuple
from dadapy.data import Data
from anatome.similarity import svcca_distance


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
        unique_data1 = torch.tensor(unique_data1, dtype=torch.float32)
        matched_data2 = torch.tensor(matched_data2, dtype=torch.float32)
        unique_data1, matched_data2 = _ensure_device(unique_data1, matched_data2)
        return unique_data1, matched_data2
    else:
        return unique_data1, matched_data2


def compute_neighborhood_overlap(
    data1: Union[np.ndarray, torch.Tensor],
    data2: Union[np.ndarray, torch.Tensor],
    maxk: int = 30,
    use_dadapy: bool = False,
    metric: str = "euclidean",
) -> float:
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
        if X_tensor.device.type == "cuda":
            indicator_X = torch.zeros((N, N), dtype=torch.int16, device=X_tensor.device)
            indicator_Y = torch.zeros((N, N), dtype=torch.int16, device=X_tensor.device)
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


# https://github.com/lorenzobasile/IDCorrelation/blob/main/utils/metrics.py
def compute_distance_correlation(
    latent: Union[np.ndarray, torch.Tensor], control: Union[np.ndarray, torch.Tensor]
) -> float:
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
    X: Union[np.ndarray, torch.Tensor],
    Y: Union[np.ndarray, torch.Tensor],
    accept_rate: float = 0.95,
) -> float:
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

    similarity = 1 - svcca_distance(X, Y, accept_rate=accept_rate, backend="svd")
    return similarity.item()


def compute_linear_cka_similarity(
    X: Union[np.ndarray, torch.Tensor], Y: Union[np.ndarray, torch.Tensor]
) -> float:
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

    return cka(X, Y, hsic=linear_hsic).item()


def compute_rbf_cka_similarity(
    X: torch.Tensor, Y: torch.Tensor, sigma: float = None
) -> float:
    """
    Compute RBF CKA similarity between two datasets.

    Args:
        X: First dataset/representation
        Y: Second dataset/representation
        sigma: Sigma for the RBF kernel

    Returns:
        RBF CKA similarity score between the two datasets
    """
    X, Y = _preprocess_data(X, Y, output_format="tensor", remove_duplicates=False)

    # Ensure tensors are on GPU if available
    X, Y = _ensure_device(X, Y)
    return cka(X, Y, hsic=kernel_hsic, sigma=sigma).item()


# https://github.com/lorenzobasile/IDCorrelation/blob/main/utils/metrics.py
def cka(
    space1: torch.Tensor,
    space2: torch.Tensor,
    hsic: callable,
    sigma: float = None,
    tolerance=1e-6,
) -> torch.Tensor:
    assert space1.shape[0] == space2.shape[0], (
        "X and Y must have the same number of samples."
    )

    numerator = hsic(space1, space2, sigma)

    var1 = torch.sqrt(hsic(space1, space1, sigma))
    var2 = torch.sqrt(hsic(space2, space2, sigma))

    cka_result = numerator / (var1 * var2)

    assert 0 - tolerance <= cka_result <= 1 + tolerance, (
        "CKA value must be between 0 and 1."
    )

    return cka_result


# https://github.com/lorenzobasile/IDCorrelation/blob/main/utils/metrics.py
def linear_hsic(X: torch.Tensor, Y: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Compute HSIC for linear kernels.

    This method is used in the computation of linear CKA.

    Args:
        X: shape (N, D), first embedding matrix.
        Y: shape (N, D'), second embedding matrix.

    Returns:
        The computed HSIC value.
    """
    # inter-sample similarity matrices for both spaces ~(N, N)
    L_X = X @ X.T
    L_Y = Y @ Y.T

    return torch.sum(center_kernel_matrix(L_X) * center_kernel_matrix(L_Y))


# https://github.com/lorenzobasile/IDCorrelation/blob/main/utils/metrics.py
def kernel_hsic(X: torch.Tensor, Y: torch.Tensor, sigma) -> torch.Tensor:
    """Compute HSIC (Hilbert-Schmidt Independence Criterion) for RBF kernels.

    This is used in the computation of kernel CKA.

    Args:
        X: shape (N, D), first embedding matrix.
        Y: shape (N, D'), second embedding matrix.
        sigma: The RBF kernel width.

    Returns:
        The computed HSIC value.
    """
    return torch.sum(
        center_kernel_matrix(rbf(X, sigma)) * center_kernel_matrix(rbf(Y, sigma))
    )


# https://github.com/lorenzobasile/IDCorrelation/blob/main/utils/metrics.py
def center_kernel_matrix(K: torch.Tensor) -> torch.Tensor:
    """Center the kernel matrix K using the centering matrix H = I_n - (1/n) 1 * 1^T. (Eq. 3 in the paper).

    This method is used in the calculation of HSIC.

    Args:
        K: The kernel matrix to be centered.

    Returns:
        The centered kernel matrix.
    """
    n = K.shape[0]
    unit = torch.ones([n, n]).type_as(K)
    identity_mat = torch.eye(n).type_as(K)
    H = identity_mat - unit / n

    return H @ K @ H


# https://github.com/lorenzobasile/IDCorrelation/blob/main/utils/metrics.py
def rbf(X: torch.Tensor, sigma=None) -> torch.Tensor:
    """Compute the RBF (Radial Basis Function) kernel for a matrix X.

    If sigma is not provided, it is computed based on the median distance.

    Args:
        X: The input matrix (num_samples, embedding_dim).
        sigma: Optional parameter to specify the RBF kernel width.

    Returns:
        The RBF kernel matrix.
    """
    GX = X @ X.T
    KX = torch.diag(GX).type_as(X) - GX + (torch.diag(GX) - GX).T

    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = math.sqrt(mdist)

    KX *= -0.5 / (sigma * sigma)
    KX = torch.exp(KX)

    return KX


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
        "rbf_cka",
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
        accept_rate = kwargs.get("accept_rate", 0.95)
        return compute_svcca_similarity(data1, data2, accept_rate)

    elif measure == "linear_cka":
        return compute_linear_cka_similarity(data1, data2)

    elif measure == "rbf_cka":
        sigma = kwargs.get("sigma", None)
        return compute_rbf_cka_similarity(data1, data2, sigma)


def compute_intrinsic_dimension(
    X: torch.Tensor,
    algorithm: str = "twoNN",
    fraction: float = 0.9,
    full_output: bool = False,
    range_max: int = 100,
    k: int = 100,
) -> float:
    """
    Compute ID (Intrinsic Dimension) of a dataset.

    Args:
        X: Dataset/representation
        algorithm: Algorithm to use for ID estimation
        fraction: Fraction of points to consider for twoNN algorithm
        full_output: Whether to return the full output or just the mean
        range_max: Maximum neighbor rank to consider for scaling_gride algorithm
        k: Number of nearest neighbors rank to consider for MLE/scaling_gride algorithm
    """
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
        X, _ = np.unique(X, axis=0, return_index=True)
        X = torch.tensor(X, dtype=torch.float32)

    if algorithm == "twoNN":
        return twoNN(X, fraction).item()
    elif algorithm == "MLE":
        return MLE(X, k, full_output).item()
    elif algorithm == "scaling_gride":
        return id_scaling_gride(X, k, range_max).item()


# https://github.com/lorenzobasile/IDCorrelation/blob/main/utils/intrinsic_dimension.py
def MLE(X: torch.Tensor, k: int = 100, full_output: bool = False) -> torch.Tensor:
    """
    Compute MLE (Maximum Likelihood Estimation) intrinsic dimension.

    Args:
        X: Dataset/representation
        k: Number of nearest neighbors to consider
        full_output: Whether to return the full output or just the mean

    Returns:
        Intrinsic dimension score between the two representations
    """
    X = X.float()
    X = _ensure_device(X)
    X = torch.cdist(X, X)
    Y = torch.topk(X, k + 1, dim=1, largest=False)[0][:, 1:]
    mask = (Y != 0).all(dim=1)
    Y = Y[mask]
    Y = torch.log(torch.reciprocal(torch.div(Y, Y[:, -1].reshape(-1, 1))))
    dim = torch.reciprocal(1 / (k - 1) * torch.sum(Y, dim=1))
    return dim if full_output else dim.mean()


# https://github.com/lorenzobasile/IDCorrelation/blob/main/utils/intrinsic_dimension.py
def twoNN(
    X: torch.Tensor, fraction: float = 0.9, distances: bool = False
) -> torch.Tensor:
    """
    Compute twoNN (Two-Nearest-Neighbors) intrinsic dimension.

    Args:
        X: Dataset/representation
        fraction: Fraction of points to consider for the linear regression
        distances: Whether to compute distances between points

    Returns:
        Intrinsic dimension score between the two representations
    """
    X = _ensure_device(X)
    if not distances:
        X = torch.cdist(X, X)
    Y = torch.topk(X, 3, dim=1, largest=False)[0]
    # clean data
    k1 = Y[:, 1]
    k2 = Y[:, 2]
    # remove zeros and degeneracies (k1==k2)
    old_k1 = k1
    k1 = k1[old_k1 != 0]
    k2 = k2[old_k1 != 0]
    old_k1 = k1
    k1 = k1[old_k1 != k2]
    k2 = k2[old_k1 != k2]
    # n.of points to consider for the linear regression
    npoints = int(np.floor(len(k1) * fraction))
    # define mu and Femp
    N = len(k1)
    mu, _ = torch.sort(torch.divide(k2, k1).flatten())
    Femp = (torch.arange(1, N + 1, dtype=X.dtype)) / N
    # take logs (leave out the last element because 1-Femp is zero there)
    x = torch.log(mu[:-1])[0:npoints]
    y = -torch.log(1 - Femp[:-1])[0:npoints]
    x, y = _ensure_device(x, y)
    slope = torch.linalg.lstsq(x.unsqueeze(-1), y.unsqueeze(-1))
    return slope.solution.squeeze()


def id_scaling_gride(
    X: torch.Tensor, k: int = 16, range_max: int = 100
) -> torch.Tensor:
    """
    Compute ID scaling grid.

    Args:
        X: Dataset/representation
        k: Density estimation scale used for ID estimation.
        range_max: Maximum neighbor rank to consider for ID estimation.
                      Creates scales {1,2,4,8,...,2^floor(log2(range_max))}.
                      Default 64 provides 7 scales: {1,2,4,8,16,32,64}.
                      Larger values give more scales but increased computation.
    """
    X = X.float()
    X = _ensure_device(X)
    distances = torch.cdist(X, X).cpu().numpy()
    data = Data(distances=distances)
    ids, _, _ = data.return_id_scaling_gride(range_max=range_max)
    id_index = int(math.log2(k))
    data.set_id(ids[id_index])
    return torch.tensor(data.intrinsic_dim, dtype=torch.float32)


def compute_renyi_entropy(
    Z: torch.Tensor, alpha: float = 1.0, eps: float = 1e-12
) -> float:
    """
    Computes the Rényi entropy of order `alpha` for a matrix Z ∈ ℝ^{N × D}

    Args:
        Z (torch.Tensor): Input matrix of shape (N, D), where N = samples and D = features.
        alpha (float): Rényi entropy order (α = 1 for Shannon).
        eps (float): Small constant for numerical stability.

    Returns:
        Scalar entropy.
    """
    # Convert to numpy and remove duplicates
    Z = Z.cpu().numpy()
    Z, _ = np.unique(Z, axis=0, return_index=True)
    Z = torch.tensor(Z, dtype=torch.float32)

    # Move to GPU if available
    Z = _ensure_device(Z)

    # Compute the Gram matrix efficiently
    N, D = Z.shape
    if N <= D:
        K = Z @ Z.T  # N x N
    else:
        K = Z.T @ Z  # D x D

    # Convert to double precision for eigendecomposition
    K = K.double()

    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(K)  # sorted, real, symmetric

    # Clamp negative eigenvalues due to numerical noise (due to numerical precision)
    eigenvalues = torch.clamp(eigenvalues, min=0.0)

    # Normalize the eigenvalues to form a probability distribution
    trace = torch.clamp(eigenvalues.sum(), min=eps)
    p = eigenvalues / trace

    # Compute entropy
    if abs(alpha - 1.0) < 1e-3:
        entropy = -torch.sum(p * torch.log(p + eps))
    else:
        entropy = torch.log(torch.sum(p**alpha) + eps) / (1 - alpha)

    return entropy.item()
