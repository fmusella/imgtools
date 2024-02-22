import numpy as np
from alabtools.utils import Index, get_index_mappings
from ...scf import SingleCellFeature
from ...scf import scf_utils


def simulate_rt(scf: SingleCellFeature) -> np.ndarray:
    """ Simulates the Replication Timing (RT) from the SingleCellFeature object.
    
    The RT is computed as the S phase profile divided by the detection bias.
    The detection bias is computed as the average of the G1 and G2 profiles.

    Args:
        scf (SingleCellFeature)

    Returns:
        rt (np.ndarray): 1D haploid RT profile.
    """

    # Assert that the cell states are defined and correspond to G1, S and G2
    if 'cell_states' not in scf:
        raise ValueError("Cell cycle states are not defined. Cannot simulate RT.")
    if not np.all(np.isin(scf.cell_states, ['G1', 'S', 'G2'])):
        raise ValueError("Cell cycle states must be 'G1', 'S' or 'G2'.")
    
    # Assert that there is a 'spot_count' matrix
    if not 'spot_count' in scf:
        raise ValueError("The spot count matrix is not defined. Cannot simulate RT.")
    
    # Calculate the bias in G1 and G2
    bias = get_bias(scf.get_matrix('spot_count'), scf.cell_states)
    
    # Get the simul ted RT as the S phase profile divided by the bias
    rt, _ = scf.haploid_profile('spot_count', isolate_state='S') / bias
    
    return rt

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


def get_bias(ncount: np.array, cycle: np.array) -> np.ndarray:
    """ Computes the detection bias from the spot counts and the cell cycle states.
    
    NOTE ON THE BIAS:
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
        ncount (np.array(ncell, ndomain, ncopy_max), dtype=int): raw single-cell spot counts.
        cycle (np.array(ndomain), dtype='U10'): cell cycle (G1, S, G2) array.

    Returns:
        bias (np.array(ndomain), dtype=float): bias array.
    """
    
    # Isolate G1 and G2 raw spots    
    ncount_g1 = ncount[cycle == 'G1', :, :]
    ncount_g2 = ncount[cycle == 'G2', :, :]
    
    # Stack the counts to get an array of shape (nG1+nG2, ndomain, ncopy_max)
    ncount_g1g2 = np.vstack((ncount_g1, ncount_g2))
    
    # Sum off the third axis, to get an array of shape (nG1+nG2, ndomain)
    ncount_g1g2 = np.nansum(ncount_g1g2, axis=2)
    
    # Normalize the counts so that each cell has the same mean = 1
    row_mean = np.nanmean(ncount_g1g2, axis=1)
    ncount_g1g2_norm = ncount_g1g2 / row_mean[:, np.newaxis]
    
    # Get the bias as the mean of the normalized counts, getting an array of shape (ndomain,)
    bias = np.nanmean(ncount_g1g2_norm, axis=0)

    return bias
