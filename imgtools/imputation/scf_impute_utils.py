import numpy as np
from alabtools.utils import Index

def impute_scf_trace_arr(arr: np.ndarray, index: Index) -> np.ndarray:
    """ Interpolate a feature array of a trace data for the SingleCellFeature data structure.
    
    The input array has missing values in the form of NaNs,
    and the function returns a new array with the missing values imputed.
    
    The code applies a simple 3D linear interpolation:
    - If the missing domain is either at the beginning or the end of the chromosome,
        the domain value is assigned as the one of the closest imaged domain.
    - If the missing domain is between two imaged pnes, the domain value is interpolated
        as the weighted average of the two closest imaged domains
        (weights are inversely proportional to the genomic distance).

    Args:
        arr (np.ndarray): trace data with missing values as NaNs. shape: (n_domains,)
        index (Index)

    Returns:
        np.ndarray: imputed trace data. shape: (n_domains,)
    """
    
    # Initialize the imputed array as a copy of the original one
    arr_imp = np.copy(arr)
    
    # Loop over the index
    for i in range(len(arr)):
        
        # If the i-th domain is not NaN, continue (no need to impute)
        if not np.isnan(arr[i]):
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
        
        # Find the neighboring domains
        left, right = find_neighbors(i, arr, index)
        
        # If both neighbors are None, something went wrong. Raise an error
        if left is None and right is None:
            raise ValueError("Error: no neighbors found for domain")
        
        # If left is None, assign the domain value as the right neighbor one
        if left is None:
            arr_imp[i] = arr[right]
        # If right is None, assign the domain value as the left neighbor one
        elif right is None:
            arr_imp[i] = arr[left]
        # Otherwise, interpolate the domain value between the two neighbors
        else:
            arr_imp[i] = linear_interpolation(i, left, right, arr, index)
    
    return arr_imp

def find_neighbors(i: int, arr: np.ndarray, index: Index) -> tuple:
    """ Find the closest imaged domains to the left and to the right of the given domain position.
    
    If either left or right neighbors are not found, they are set to None.

    Args:
        i (int): domain position of interest
        arr (np.ndarray): trace data with missing values as NaNs. shape: (n_domains,)
        index (Index)

    Returns:
        (int, int): left and right neighbor positions (or None if not found)
    """
    
    # Find the closest imaged domain to the left
    left = None
    for j in range(i - 1, -1, -1):
        if not np.isnan(arr[j]):
            left = j
            break
    
    # Find the closest imaged domain to the right
    right = None
    for j in range(i + 1, len(index)):
        if not np.isnan(arr[j]):
            right = j
            break
    
    return left, right

def linear_interpolation(i: int, l: int, r: int, arr: np.ndarray, index: Index) -> float:
    """ Perform a linear interpolation between two imaged domains.
    
    The interpolation is weighted by the inverse of the genomic distance between the domains:
       w_ir = 1 - |s[i] - s[r]| / (|s[i] - s[r]| + |s[i] - s[l]|)
       w_il = 1 - |s[i] - s[l]| / (|s[i] - s[r]| + |s[i] - s[l]|)
    where s[i] is the genomic position of the i-th domain.
    
    So, if for example the genomic distance between i and r is 0, the weights are w_r = 1 and w_l = 0,
    so the domain value is assigned as the one of the right neighbor.
    
    The weighted average is then computed as:
        w_r * arr[r] + w_l * arr[l]

    Args:
        i (int): domain position of interest
        l (int): left neighbor position
        r (int): right neighbor position
        arr (np.ndarray): trace data with missing values as NaNs. shape: (n_domains,)
        index (Index)

    Returns:
        float: interpolated i domain value
    """
    
    # Get the genomic distances from i to the left and right neighbors
    gendist_ir = np.abs(index.start[i] - index.start[r])
    gendist_il = np.abs(index.start[i] - index.start[l])
    
    # Create weights for the interpolation: the closer the genomic distance, the higher the weight
    w_r = 1. - gendist_ir / (gendist_ir + gendist_il)
    w_l = 1. - gendist_il / (gendist_ir + gendist_il)
    
    # Interpolate the domain value by weighting the neighbors values
    return w_r * arr[r] + w_l * arr[l]
