import numpy as np
from alabtools.utils import Index, get_index_sliding_mapping
from ..scf import SingleCellFeature
from ..scf import scf_utils


def simulate_rt(
    scf: SingleCellFeature,
    feature: str,
    normalize: bool = True,
    resolution: int = None
) -> np.ndarray:
    """ Simulates the Replication Timing (RT) from the SingleCellFeature object.
    
    It uses the feature matrix of the input feature from the SCF object:
    1. It measures a 1D bias from cells in G1 and G2.
    2. It computes the 1D RT profile from cells in S phase and divides it by the bias.

    Args:
        scf (SingleCellFeature)
        feature (str): feature name to simulate RT from.

    Returns:
        rt (np.ndarray): 1D RT profile.
    """

    # Assert that the cell states are defined and there is an S phase
    if 'cell_states' not in scf:
        raise ValueError("Cell cycle states are not defined. Cannot simulate RT.")
    if not np.any(scf.cell_states == 'S'):
        raise ValueError("There is no S phase. Cannot simulate RT.")
    
    # Assert that the feature is in the SingleCellFeature object
    if not feature in scf:
        raise ValueError(f"The feature {feature} is not in the SingleCellFeature object.")
    
    # Get the matrix of the input feature
    mat = scf.get_matrix(feature)
    
    # Set the 0s to NaNs
    mat = mat.astype(np.float32)
    mat[mat == 0] = np.nan
    
    # Coarse-grain the matrix if a resolution is given
    if resolution is not None:
        mat, _ = scf_utils.coarsegrain_matrix(mat, scf.index, resolution, method='average')
    
    # Get the S phase profile
    rt = np.nanmean(mat[scf.cell_states == 'S', :, :], axis=(0, 2))
    
    # Normalize by the bias
    if normalize:
        # Calculate the bias in G1 and G2
        bias = get_bias(mat, scf.cell_states)
        # Normalize the S phase profile by the bias
        rt = rt / bias
    
    return rt


def simulate_replication(scf: SingleCellFeature, resolution: int) -> np.ndarray:
    """ PROVISONIAL FUNCTION, only uses spot count"""
    
    # Load the spot count matrix
    spotcount = scf.get_matrix('spotcount')
    
    # Load the index
    index = scf.index
    
    # Initialize the replication matrix
    replication = np.ones(spotcount.shape).astype(np.float32)
    
    # Get the sliding index mapping
    sliding_mapping = get_index_sliding_mapping(index, int(resolution / index.resolution()))
    
    # Loop over the genomic domains
    for i in range(len(index)):
        
        # Get the indices of the bins that are included in the sliding window
        indices = sliding_mapping[i]
        
        # Get the data for these indices
        spotcount_i = spotcount[:, indices, :]  # ncells x window x ncopies
        
        # Get a matrix nan_mat of shape ncells x ncopies such that
        #   nan_mat[i, j] = True if all values in spotcount_i[i, :, j] are nan
        #   nan_mat[i, j] = False otherwise
        is_nan_mat = np.all(np.isnan(spotcount_i), axis=1)
        
        # Get a matrix 2_mat of shape ncells x ncopies such that
        #   2_mat[i, j] = True if any value in spotcount_i[i, :, j] is >= 2
        #   2_mat[i, j] = False otherwise
        is_two_mat = np.any(spotcount_i >= 2, axis=1)
        
        # Assert that there is no overlap between the two matrices
        assert np.all(np.logical_not(np.logical_and(is_nan_mat, is_two_mat))), "Overlap between nan_mat and two_mat."
        
        # Get the replication matrix for this domain
        replication_i = np.ones(is_nan_mat.shape).astype(np.float32)
        replication_i[is_nan_mat] = np.nan
        replication_i[is_two_mat] = 2
        
        # Assign the replication matrix to the output matrix
        replication[:, i, :] = replication_i
        
    return replication


def simulate_singlecell_replication(scf: SingleCellFeature, resolution: int) -> np.ndarray:
    """ Simulates the single-cell replication matrix from the SingleCellFeature object.
    The SCF must contain the spot count matrix and the cell cycle states.
    The spot count matrix has the shape (ncell, ndomain, ncopy_max).

    Args:
        scf (SingleCellFeature)
        resolution (int): window resolution in bp for the sliding window sum.

    Returns:
        np.ndarray: replication matrix of the same shape as the spot count matrix.
    """
    
    # Assert that the cell states are defined and correspond to G1, S and G2
    if 'cell_states' not in scf:
        raise ValueError("Cell cycle states are not defined. Cannot simulate RT.")
    if not np.all(np.isin(scf.cell_states, ['G1', 'S', 'G2'])):
        raise ValueError("Cell cycle states must be 'G1', 'S' or 'G2'.")
    
    # Assert that there is a 'spot_count' matrix
    if not 'spot_count' in scf:
        raise ValueError("The spot count matrix is not defined. Cannot simulate RT.")
    
    # Assert that the resolution is a multiple of the index resolution
    if resolution % scf.index.resolution() != 0:
        raise ValueError("The resolution must be a multiple of the index resolution.")
    # Assert that the resolution is an odd multiple of the index resolution
    if (resolution // scf.index.resolution()) % 2 == 0:
        raise ValueError("The resolution must be an odd multiple of the index resolution.")
    
    # Calculate the bias in G1 and G2
    bias = get_bias(scf.get_matrix('spot_count'), scf.cell_states)  # (ndomain,)
    
    # Get the normalized spot count matrix
    n = scf.get_matrix('spot_count')  # (ncell, ndomain, ncopy_max)
    nu = n / bias[np.newaxis, :, np.newaxis]  # (ncell, ndomain, ncopy_max)
    
    # Binarize the normalized spot count matrix
    thresh_1 = 1.08
    nu_bin = np.zeros(nu.shape, dtype=np.int32)
    nu_bin[nu > thresh_1] = 1
    
    # Perform a sliding window sum to the binarized matrix
    window = int(resolution // scf.index.resolution())
    nu_bin_windowsum = scf_utils.sliding_matrix(nu_bin, scf.index, window=window, method='sum')
    
    # Binarize the window-summed matrix, getting the replication matrix
    thresh_2 = 2
    rho = np.ones(nu_bin_windowsum.shape, dtype=np.int32)
    rho[nu_bin_windowsum > thresh_2] = 2
    
    # Get the significance matrix, which tell us - for each domain - if we can trust the replication matrix
    # First binarize the spot count matrix
    n_bin = np.zeros(n.shape, dtype=np.int32)
    n_bin[n > 0] = 1
    # Secondly, remove outliers from the spot count matrix, i.e. windows with a very high number of spots
    outlier_thresh = 4
    # n_bin[n > outlier_thresh] = 0  # set outliers to 0
    # Then, perform a sliding window sum to the number of spots
    n_bin_windowsum = scf_utils.sliding_matrix(n_bin, scf.index, window=window, method='sum')
    # We trust only those regions where more than a given percentage of the window size have spots
    perc = 0.5
    confidence_thresh = int(np.floor(perc * window))  # perc of the window size
    sig = np.zeros(n_bin_windowsum.shape, dtype=np.int32)
    sig[n_bin_windowsum > confidence_thresh] = 1
    
    return rho, sig


def get_bias(matrix: np.array, states: np.array) -> np.ndarray:
    """ Computes the G1/G2 bias of an input feature matrix.
    
    NOTE ON THE BIAS:
    This is easier to understand if we think of the feature matrix as the spot count matrix.
    Since the cells in G1 and G2 are not replicating, variation in the total number of spots is due noise or bias.
    If we see that a domain has systematically more/less spots than others in G1 or G2,
    we can assume that this is due to bias and not noise
    (for example GC rich domains are detected more likely than AT rich domains).
    Therefore, we can estimate the bias by computing the total number of spots
    in each domain in G1 and G2.
    To weigh each cell in G1/G2 equally, we normalize the counts so that each cell has the same mean = 1.
    This is relevant for two reasons:
        1) G2 have double the DNA content of G1, so they would be weighted twice as much.
        2) the detection efficiency of each cell could be different, so high-efficiency cells would be weighted more.

    Args:
        matrix (np.array(ncell, ndomain, ncopy_max), dtype=int): feature matrix, un-normalized.
        states (np.array(ndomain), dtype='U10'): cell cycle states (G1, S, G2) array.

    Returns:
        bias (np.array(ndomain), dtype=float): bias array.
    """
    
    # Isolate G1 and G2 feature submatrix 
    matrix_g1g2 = matrix[states != 'S', :, :]
    
    # Sum off the third axis, to get an array of shape (nG1+nG2, ndomain)
    matrix_g1g2 = np.nansum(matrix_g1g2, axis=2)
    
    # Normalize the rows so that each cell has the same mean = 1
    row_mean = np.nanmean(matrix_g1g2, axis=1)
    matrix_g1g2_norm = matrix_g1g2 / row_mean[:, np.newaxis]
    
    # Get the bias as the mean of the normalized rows, getting an array of shape (ndomain,)
    bias = np.nanmean(matrix_g1g2_norm, axis=0)

    return bias
