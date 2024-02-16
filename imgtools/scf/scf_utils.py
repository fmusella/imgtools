import numpy as np
from alabtools.utils import Index, get_index_mappings


def coarsegrain_matrix(mat: np.ndarray, index: Index, resolution: int, method: str) -> tuple:
    """ Coarse-grain a feature matrix to a specified resolution.
    The key 'method' specifies how the high-resolution data is coarse-grained to the low-resolution bins.
    Available methods are:
    - 'average': the average value of the high-resolution bins is assigned to the low-resolution bin.
    - 'sum': the sum of the high-resolution bins is assigned to the low-resolution bin.
    - 'consensus': the low-resolution bin is assigned 1 if the majority of the high-resolution bins are 1 or more.

    Args:
        mat (np.ndarray): feature matrix of shape ncells x ndomains x ncopies.
        index (Index): index of the feature matrix.
        resolution (int): low-resolution for the coarse-grained matrix.
        method (str): method to coarse-grain the data. Available methods are 'average', 'sum', and 'consensus'.

    Returns:
        (np.ndarray): coarse-grained feature matrix of shape ncells x ndomains_coarse x ncopies.
        (Index): coarse-grained index at the specified resolution.
    """
    
    # Get the coarse-grained index
    index_coarse = index.coarsegrain(resolution)
    
    # Initialize the matrix to store the coarse-grained data
    mat_coarse = np.zeros((mat.shape[0], len(index_coarse), mat.shape[2]), dtype=mat.dtype)
    
    # Get mappings to coarse-grain the signals in the index
    _, _, bmap = get_index_mappings(index, index_coarse)
    
    # Loop over the bins of the coarse index
    for i in range(len(index_coarse)):
        
        # Get the indices of the fine-grain bins that are included in the coarse-grain bin i
        indices = bmap[i]
        
        # Get the high-resolution data for these indices
        mat_indices = mat[:, indices, :]
        
        # Use the specified method to coarse-grain the data
        
        if method == 'average':
            mat_coarse[:, i, :] = np.nanmean(mat_indices, axis=1)
        
        elif method == 'sum':
            mat_coarse[:, i, :] = np.nansum(mat_indices, axis=1)
        
        elif method == 'consensus':
            # We assign 1 if the majority of the high-resolution bins are 1 (more than 50% of the bins are 1 or more)
            mat_coarse_sum = np.nansum(mat_indices, axis=1)
            mat_coarse[:, i, :] = (mat_coarse_sum > (len(indices) / 2)).astype(np.float32)
        
        else:
            raise ValueError(f"Method {method} not recognized.")
    
    return mat_coarse, index_coarse


def normalize_matrix(mat: np.ndarray, norm_arr: np.ndarray = None,  by_zscore: bool = False) -> np.ndarray:
    """ Normalize a feature matrix.
    
    Available normalization methods are:
    - by a normalization array: the feature matrix is divided by the normalization array, which has the same length as the number of cells.
    - by z-scoring: the feature matrix is z-scored in each cell.
    
    Multiple normalization methods can be applied at the same time, with the order of application being the following:
      1) by a normalization array, 2) by z-scoring.

    Args:
        mat (np.ndarray): feature matrix of shape ncells x ndomains x ncopies.
        norm_arr (np.ndarray, optional): normalization array of shape ncells. Defaults to None.
        by_zscore (bool, optional): if True, the feature matrix is z-scored. Defaults to False.

    Returns:
        (np.ndarray): normalized feature matrix of shape ncells x ndomains x ncopies.
    """
    if norm_arr is not None:
        if not len(norm_arr) == mat.shape[0]:
            raise ValueError("The length of the normalization array must be equal to the number of cells.")
        mat = mat / norm_arr[:, np.newaxis, np.newaxis]
    if by_zscore:
        # z-score the matrix in each cell
        mean = np.nanmean(mat, axis=(1, 2))[:, np.newaxis, np.newaxis]
        std = np.nanstd(mat, axis=(1, 2))[:, np.newaxis, np.newaxis]
        mat = (mat - mean) / std
    return mat
