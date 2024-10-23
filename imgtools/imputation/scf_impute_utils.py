import numpy as np

def impute_scf_trace_data(feats: np.ndarray, genpos: np.ndarray) -> np.ndarray:
    """ Interpolate a feature data array of a chromosomal trace feature data
    for the SingleCellFeature data structure.
    
    The input array has missing values in the form of NaNs,
    and the function returns a new array with the missing values imputed.
    
    The code applies a simple 3D linear interpolation:
    - If the missing domain is either at the beginning or the end of the chromosome,
        the domain value is assigned as the one of the closest imaged domain.
    - If the missing domain is between two imaged ones, the domain value is interpolated
        as the weighted average of the two closest imaged domains
        (weights are inversely proportional to the genomic distance).

    Args:
        feats (np.ndarray): feature values of the trace with missing values as NaNs. shape: (n_domains,)
        genpos (np.ndarray): genomic positions of the domains. shape: (n_domains,)

    Returns:
        np.ndarray: imputed trace data. shape: (n_domains,)
    """
    
    # Initialize the imputed feature array as a copy of the original one
    feats_imp = np.copy(feats)
    
    # Get the array positions of the imaged domains, i.e. the non-NaN values
    # e.g. [0, 42, 103, ...]
    # where the values are the positions of the imaged domains in the array
    js_imgd = np.where(~np.isnan(feats))[0]
    
    # Loop over the domain positions in the array
    # Note: as opposed to the imputation code for the ChromatinTracingExperiment,
    # here we are looping directly over the positions of the trace domains,
    # so there is no issue of spanning over other chromosomes
    for i in range(len(feats)):
        
        # If the i-th domain is not NaN, continue (no need to impute)
        if not np.isnan(feats[i]):
            continue
        
        # Otherwise, we need to interpolate the spot data
        # There are three cases:
        #  1. The domain is at the beginning of the chromosome,
        #     i.e. there are no imaged domains to the left
        #  2. The domain is at the end of the chromosome,
        #     i.e. there are no imaged domains to the right
        #  3. The domain is between two imaged domains
        # In the first two cases we simply assign the domain value as the one of the closest imaged domain
        # In the third case we interpolate the domain value between the two closest imaged domains
        
        # Find the neighboring domains' positions
        l, r = find_neighbors(i, js_imgd)
        
        # If both neighbors are None, something went wrong. Raise an error
        if l is None and r is None:
            raise ValueError("Error: no neighbors found for domain")
        
        # If left is None, assign the feature value of the right neighbor
        if l is None:
            feats_imp[i] = feats[r]
        # If right is None, assign the feature value of the left neighbor
        elif r is None:
            feats_imp[i] = feats[l]
        # Otherwise, interpolate the feature value between the two neighbors
        else:
            feats_imp[i] = linear_interpolation(i, l, r, feats, genpos)
    
    return feats_imp

def find_neighbors(i: int, js_imgd: np.ndarray) -> tuple:
    """ Find the closest imaged domains to the left and to the right of the given domain position.
    
    If either left or right neighbors are not found, they are set to None.

    Args:
        i (int): Position of the domain for which to find neighbors.
        js_imgd (np.ndarray): Array of the imaged domain positions. shape = (n_imaged_domains,)

    Returns:
        tuple: The left and right neighbors of the domain.
    """

    # Initialize the neighbors to None
    l = None
    r = None
    
    # Calculate the differences between the imaged domain positions and the current domain
    diffs = js_imgd - i
    
    # Split the differences into those to the left of i (negative) and those to the right of i (positive)
    mask = diffs < 0
    diffs_l = diffs[mask]
    diffs_r = diffs[~mask]
    
    # If there are imaged domains to the left, get the closest one
    if len(diffs_l) > 0:
        l = diffs_l.max() + i
    
    # If there are imaged domains to the right, get the closest one
    if len(diffs_r) > 0:
        r = diffs_r.min() + i
    
    return l, r

def linear_interpolation(i: int, l: int, r: int, feats: np.ndarray, genpos: np.ndarray) -> float:
    """ Perform a linear interpolation between two imaged domains.
    
    The interpolation is weighted by the inverse of the genomic distance between the domains:
       w_ir = 1 - |s[i] - s[r]| / (|s[i] - s[r]| + |s[i] - s[l]|)
       w_il = 1 - |s[i] - s[l]| / (|s[i] - s[r]| + |s[i] - s[l]|)
    where s[i] is the genomic position of the i-th domain.
    
    So, if for example the genomic distance between i and r is 0, the weights are w_r = 1 and w_l = 0,
    so the domain value is assigned as the one of the right neighbor.
    
    The weighted average is then computed as:
        w_r * feats[r] + w_l * feats[l]

    Args:
        i (int): domain position of interest
        l (int): left neighbor position
        r (int): right neighbor position
        feats (np.ndarray): feature values of the trace. shape: (n_domains,)
        genpos (np.ndarray): genomic positions of the domains. shape: (n_domains,)

    Returns:
        float: interpolated i domain value
    """
    
    # Get the genomic distances from i to the left and right neighbors
    gendist_ir = np.abs(genpos[i] - genpos[r])
    gendist_il = np.abs(genpos[i] - genpos[l])
    
    # Create weights for the interpolation: the closer the genomic distance, the higher the weight
    w_r = 1. - gendist_ir / (gendist_ir + gendist_il)
    w_l = 1. - gendist_il / (gendist_ir + gendist_il)
    
    # Interpolate the domain value by weighting the neighbors values
    return w_r * feats[r] + w_l * feats[l]
