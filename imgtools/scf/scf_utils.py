import numpy as np
from alabtools.utils import Index, map_indices, get_index_sliding_mapping


def coarsegrain_matrix(mat: np.ndarray, index: Index, resolution, method: str) -> tuple:
    """ Coarse-grain a feature matrix to a specified resolution.
    The key 'method' specifies how the high-resolution data is coarse-grained to the low-resolution bins.
    Available methods are:
    - 'average': the average value of the high-resolution bins is assigned to the low-resolution bin.
    - 'sum': the sum of the high-resolution bins is assigned to the low-resolution bin.
    - 'consensus': the low-resolution bin is assigned 1 if the majority of the high-resolution bins are 1 or more.

    Args:
        mat (np.ndarray): feature matrix of shape ncells x ndomains x ncopies.
        index (Index): index of the feature matrix.
        resolution (Index or int): corase index, or coarse resolution.
        method (str): method to coarse-grain the data. Available methods are 'average', 'sum', and 'consensus'.

    Returns:
        (np.ndarray): coarse-grained feature matrix of shape ncells x ndomains_coarse x ncopies.
        (Index): coarse-grained index at the specified resolution.
    """
    
    # Get the coarse-grained index
    index_coarse = index.coarsegrain(resolution)
    # Calculate the ratio of the resolutions
    res_ratio = int(index_coarse.resolution() / index.resolution())
    
    # Map the indices from the coarse-grained index to the high-resolution index, e.g.
    #    map_coarse_to_high = {
    #           ('chr1', 100000, 150000): [('chr1', 100000, 125000), ('chr1', 125000, 150000)],
    #           ('chr1', 150000, 200000): [('chr1', 150000, 175000), ('chr1', 175000, 200000)],
    #           ...
    #       }
    map_coarse_to_high = map_indices(index_coarse, index)
    
    # Check that the mapping is correct:    
    # 1) the length of the mapping is the same as the length of the coarse-grained index
    assert len(map_coarse_to_high) == len(index_coarse), "Length of the mapping does not match the length of the coarse-grained index."
    for domcoarse in map_coarse_to_high:
        doms = map_coarse_to_high[domcoarse]
        # 2) the number of domains that map to the coarse-grained domain is equal or less to the ratio of the coarse and original resolutions
        assert isinstance(doms, list), "The domain does not map to a list."
        assert len(doms) <= res_ratio, "The number of domains that map to the coarse-grained domain is larger than the ratio of resolutions."
        for dom in doms:
            # 3) the chromosomes of the original and coarse-grained indices match
            assert domcoarse[0] == dom[0], "Chromosomes of the original and coarse-grained indices do not match."
            # 4) The start/end of the coarse-grained domain includes the start/end of the original domain
            assert domcoarse[1] <= dom[1], "Start positions of the original and coarse-grained indices do not match."
            assert domcoarse[2] >= dom[2], "End positions of the original and coarse-grained indices do not match."
    
    # Get the hashmap of the high-resolution index, e.g.
    #  index_hashmap = {
    #       ('chr1', 100000, 125000): [0],
    #       ('chr1', 125000, 150000): [1],
    #       ...
    #   }
    index_hashmap = index.get_index_hashmap()
    
    # Initialize the matrix to store the coarse-grained data
    mat_coarse = np.zeros((mat.shape[0], len(index_coarse), mat.shape[2]), dtype=mat.dtype)
    
    # Loop over the bins of the coarse index
    for i, domcoarse in enumerate(zip(index_coarse.chromstr, index_coarse.start, index_coarse.end)):
        
        # Get the domains of the high-resolution index that are included in the coarse bin
        doms = map_coarse_to_high[domcoarse]

        # Get the high-resolution data for these indices
        indices = [index_hashmap[dom][0] for dom in doms]
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


def sliding_matrix(mat: np.ndarray, index: Index, window: int, method: str) -> np.ndarray:
    """ Apply a sliding window operation to a feature matrix.
    The key 'method' specifies how the data within the sliding window is processed.
    Available methods are 'mean', 'median', and 'sum'.

    Args:
        mat (np.ndarray): feature matrix of shape ncells x ndomains x ncopies.
        index (Index): index of the feature matrix.
        window (int): window size for the sliding operation.
        method (str): method to process the data within the sliding window.

    Returns:
        np.ndarray: sliding-window-processed feature matrix of shape ncells x ndomains x ncopies.
    """
    
    assert len(index) == mat.shape[1], "The length of the index must be equal to the number of genomic bins."
    
    # Check that the method provided is valid
    valid_methods = ['mean', 'median', 'sum']
    if not method in valid_methods:
        raise ValueError(f"Method {method} not recognized. Available methods are {valid_methods}.")
    
    # Get the sliding index mapping
    sliding_mapping = get_index_sliding_mapping(index, window)
    
    # Initialize the matrix to store the sliding data
    out_mat = np.zeros(mat.shape).astype(mat.dtype)  # ncells x ndomains x ncopies
    
    # Loop over the genomic domains
    for i in range(len(index)):
        
        # Get the indices of the bins that are included in the sliding window
        indices = sliding_mapping[i]
        
        # Get the data for these indices
        mat_i = mat[:, indices, :]  # ncells x window x ncopies
        
        # Use the specified method to process the data
        if method == 'mean':
            out_mat[:, i, :] = np.nanmean(mat_i, axis=1)
        elif method == 'median':
            out_mat[:, i, :] = np.nanmedian(mat_i, axis=1)
        elif method == 'sum':
            out_mat[:, i, :] = np.nansum(mat_i, axis=1)
    
    return out_mat


def quantize_matrix(mat: np.ndarray, nquants: int) -> np.ndarray:
    """ Quantize a feature matrix, separately for each cell, into 'nquants' quantiles.
    
    Creates a quantized version of the feature matrix: qmat: (ncells, nloci, ncopies).
    This is an int array, where each value qmat[c, i, h] is the quantized value of mat[c, i, h]
    with respect to the other values in the same cell, mat[c, :, :].

    Args:
        mat (np.ndarray): feature matrix. shape: (ncells, nloci, ncopies).
        nquants (int): number of quantiles to divide the feature data.

    Returns:
        qmat (np.ndarray): quantized feature matrix. shape: (ncells, nloci, ncopies).
        quants (np.ndarray): quantiles of the feature data. shape: (nquants).
    """
    
    # Check the shape of the input matrix, it must be (ncells, nloci, ncopies)
    try:
        ncells, _, _ = mat.shape
    except ValueError:
        raise ValueError("The input matrix must have shape (ncells, nloci, ncopies).")
    
    # Initialize the quantized feature matrix
    # We initialize with -1: the NaN values in mat will remain as -1
    qmat = np.full(mat.shape, -1, dtype=int)  # shape: (ncells, nloci, ncopies)
    
    # Loop over the cells
    for c in range(ncells):
        
        # Get the feature data for the cell
        mat_c = mat[c, :, :]  # shape: (nloci, ncopies)
        
        # Quantize the feature data for the cell
        qmat_c = quantize_matrix_cell(mat_c, nquants)
        
        # Store the quantized data for the cell
        qmat[c, :, :] = qmat_c
    
    # Get the quantiles as an array
    quants = np.arange(nquants)
    
    return qmat, quants

def quantize_matrix_cell(mat_c: np.ndarray, nquants: int) -> np.ndarray:
    """ Quantize the feature matrix of a single cell into 'nquants' quantiles.

    Args:
        mat_c (np.ndarray): feature matrix of a single cell. shape: (nloci, ncopies).
        nquants (int): number of quantiles to divide the feature data.

    Returns:
        np.ndarray: quantized feature matrix of a single cell. shape: (nloci, ncopies).
    """
    
    # Initialize the quantized data for the cell
    # We initialize with -1: the NaN values in mat will remain as -1
    qmat_c = np.full(mat_c.shape, -1, dtype=int)  # shape: (nloci, ncopies)
    
    # Get the quantiles of the cell
    quants_c = np.nanquantile(mat_c, np.linspace(0, 1, nquants + 1))  # shape: (nquants + 1)
    
    # Loop over the quantiles
    for q in range(nquants):
        # Get the mask for the quantile
        if q == nquants - 1:
            mask_q = mat_c >= quants_c[q]  # include the last value if it's the last quantile
        else:
            mask_q = np.logical_and(mat_c >= quants_c[q], mat_c < quants_c[q + 1])
        # Assign the quantile to the quantized data
        qmat_c[mask_q] = q
    
    return qmat_c


def curate_missing_chromosomes(mat: np.ndarray, index: Index) -> None:
    """ Set the entries of a matrix of shape (ncells, nloci, ncopies) to NaN
    for missing chromosomal traces.
    
    Changes the input matrix in place.

    Args:
        mat (np.ndarray): matrix of shape (ncells, nloci, ncopies).
    """
    
    # Check the shape of the input matrix, it must be (ncells, nloci, ncopies)
    try:
        ncells, _, ncopies = mat.shape
    except ValueError:
        raise ValueError("The input matrix must have shape (ncells, nloci, ncopies).")
    
    # Loop over cells
    for cellnum in range(ncells):
    
        # Loop over the chromosomes and mask them
        for chrom in index.genome.chroms:
            mask_chrom = index.chromstr == chrom  # shape: (nloci)
            
            # Loop over the copies
            for copynum in range(ncopies):
                
                # If the matrix of the cell/chrom/copy is made of only 0s, set it as NaN in the object
                if np.all(mat[cellnum, mask_chrom, copynum] == 0):
                    mat[cellnum, mask_chrom, copynum] = np.nan


def z_score_matrix(mat: np.ndarray, states: np.ndarray = None) -> np.ndarray:
    """ Z-score a feature matrix locus-wide:
            zmat[c, i, h] = (mat[c, i, h] - mean_i(mat[c, :, h])) / std_i(mat[c, :, h])
    
    The mean and std per locus are calculated as averages across cells,
    but only considering the cells in the same state (e.g. G1 cells).
    
    If no states are provided, all cells are considered to be in the same state.

    Args:
        mat (np.ndarray): feature matrix, shape: (ncells, nloci, ncopies).
        states (np.ndarray, optional): states of the cells. shape: (ncells). Defaults to None.

    Returns:
        np.ndarray: z-scored feature matrix, shape: (ncells, nloci, ncopies).
    """
    
    # Initialize the z-scored matrix
    zmat = np.zeros(mat.shape)  # shape: (ncells, nloci, ncopies)
    
    # If no states are provided, set the states as the same in each cell
    if states is None:
        states = np.ones(mat.shape[0])
    
    # Loop over the states
    for s in np.unique(states):
        
        # Get the mask for the current state
        mask_s = states == s
        
        # Calculate mean and std per locus only with the cells in the current state
        mean_s = np.nanmean(mat[mask_s, :, :], axis=(0, 2))  # shape: (nloci)
        std_s = np.nanstd(mat[mask_s, :, :], axis=(0, 2))  # shape: (nloci)
        
        # Cast the mean and std to the same shape as the matrix
        mean_s = np.tile(mean_s[np.newaxis, :, np.newaxis], (np.sum(mask_s), 1, mat.shape[2]))  # shape: (ncells_s, nloci, ncopies)
        std_s = np.tile(std_s[np.newaxis, :, np.newaxis], (np.sum(mask_s), 1, mat.shape[2]))  # shape: (ncells_s, nloci, ncopies)
        
        # Z-score the matrix for the current state and store it in the z-scored matrix
        zmat[mask_s, :, :] = (mat[mask_s, :, :] - mean_s) / std_s
    
    return zmat
