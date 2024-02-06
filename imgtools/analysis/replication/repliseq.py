import numpy as np
from ...scf import SingleCellFeature


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
    bias = normalize_bias_new(scf.get_matrix('spot_count'), scf.cell_states)
    
    # Get the simulated RT as the S phase profile divided by the bias
    rt, _ = scf.haploid_profile('spot_count', isolate_state='S') / bias
    
    return rt


def normalize_bias(ncount: np.array, cycle: np.array) -> np.array:
    """Normalize the bias in the raw spots counts.
    
    NOTE ON THE BIAS:
    Since the cells in G1 and G2 are not replicating, variation in the total number of spots is due noise or bias.
    If we see that a domain has systematically more/less spots than others in G1 or G2,
    we can assume that this is due to bias and not noise
    (for example GC rich domains are detected more likely than AT rich domains).
    Therefore, we can estimate the bias by computing the total number of spots
    in each domain in G1 and G2.

    Returns:
        bias (np.array(ndomain), dtype=float): bias array.
    """
    
    # Isolate G1 and G2 raw spots    
    ncount_g1 = ncount[cycle == 'G1', :, :]
    ncount_g2 = ncount[cycle == 'G2', :, :]
    
    # Get the bias as the sum of the spots in G1 and G2
    bias_g1 = np.nansum(ncount_g1, axis=(0, 2))  # np.array(ndomain)
    bias_g2 = np.nansum(ncount_g2, axis=(0, 2))
    
    # Rescale the bias arrays to have mean 1
    bias_g1 = bias_g1 / np.nanmean(bias_g1)
    bias_g2 = bias_g2 / np.nanmean(bias_g2)
    
    # Set the total bias as mean of the G1 and G2 biases
    bias = (bias_g1 + bias_g2) / 2
    
    # If bias_g1 has NaNs, set the bias as bias_g2 and vice versa
    bias[np.isnan(bias_g1)] = bias_g2[np.isnan(bias_g1)]
    bias[np.isnan(bias_g2)] = bias_g1[np.isnan(bias_g2)]
    
    # Rescale the bias to have mean 1
    # (again, since NaNs could have screwed up the mean)
    bias = bias / np.nanmean(bias)
    
    return bias


def normalize_bias_new(ncount: np.array, cycle: np.array) -> np.array:
    """Normalize the bias in the raw spots counts.
    
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
