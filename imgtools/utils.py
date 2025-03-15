import os
import numpy as np
from scipy.spatial import distance
from scipy.stats import pearsonr, spearmanr
from alabtools.utils import Index


def spots_3d_median(points: np.ndarray, centroid: np.ndarray) -> int:
    """ Given a list of spot points associated with the same domain in a trace,
    selects only one of them with the 3D median criterion.
    
    There are three cases:
        1) If there is only one point, return the index of that point (0)
        2) If there are two points, return the one closer to the centroid
        3) If there are three or more points, return the point that minimizes the sum of distances to all other points
    
    The function returns the index of the selected point.
    
    The computation for point 3, the actual 3D median, is based on the 3D geometric median (https://en.wikipedia.org/wiki/Geometric_median).
    However, in this reference the algorithm finds the point in 3D space that minimizes the sum of distances to all other points,
    so it doesn't return a point of the set. Here we don't want to create 'fake' points, so I adapted the algorithm to only
    consider the points in the set.
    
    Args:
        points (np.ndarray): array of shape (npoints, 3) containing the 3D coordinates of the spots.
        centroid (np.ndarray): array of shape (3,) containing the 3D coordinates of the centroid of the trace.

    Returns:
        median_idx (int): index of the selected point, between 0 and npoints-1.
    """
    
    # Check the points array
    if not isinstance(points, np.ndarray):
        raise TypeError('points must be a numpy array')
    if len(points.shape) != 2:
        raise ValueError('points must be a numpy array of shape (npoints, 3)')
    if points.shape[1] != 3:
        raise ValueError('points must be a numpy array of shape (npoints, 3)')
    if points.shape[0] == 0:
        raise ValueError('points must be a numpy array of shape (npoints, 3)')
    npoints = points.shape[0]  # get the number of points
    
    # Check the centroid array
    if not isinstance(centroid, np.ndarray):
        raise TypeError('centroid must be a numpy array')
    if centroid.shape != (3,):
        raise ValueError('centroid must be a numpy array of shape (3,)')
    
    # If there is only one point, return the index of that point (0)
    if npoints == 1:
        return 0
    
    # If there are two points, return the one closer to the centroid
    elif npoints == 2:
        median_idx = 0 if distance.euclidean(points[0], centroid) <= distance.euclidean(points[1], centroid) else 1
        return median_idx
    
    # Otherwise, find the point that minimizes the sum of distances to all other points
    # Initialize the median index and the minimum distance
    median_idx = None
    dists_min = np.inf
    # Loop over the points
    for i in range(len(points)):
        dists_i = 0  # initialize the sum of distances for point i
        for j in range(len(points)):
            if i == j:
                continue
            dists_i = dists_i + distance.euclidean(points[i], points[j])  # add the distance between point i and point j
        # If the total distance is smaller than the minimum distance, update the median index and the minimum distance
        if dists_i < dists_min:
            median_idx = i
            dists_min = dists_i
        # If the total distances are equal, choose the point closer to the centroid
        if dists_i == dists_min:
            if distance.euclidean(points[i], centroid) < distance.euclidean(points[median_idx], centroid):
                median_idx = i
                dists_min = dists_i
    return median_idx


def compare_index(idx1: Index, idx2: Index, usechr: list) -> bool:
    """Compares two Index objects on a subset of chromosomes.

    Args:
        idx1 (Index): first Index object.
        idx2 (Index): second Index object.
        usechr (list): list of chromosomes to be used in the comparison.

    Returns:
        bool: True if the two Index objects are the same on the subset of chromosomes.
    """
    
    if idx1.genome.assembly != idx2.genome.assembly:
        return False
    
    # Compare the two Index objects on the chromosomes in usechr
    if np.any(idx1.chromstr[np.isin(idx1.chromstr, usechr)] != idx2.chromstr[np.isin(idx2.chromstr, usechr)]):
        return False
    if np.any(idx1.start[np.isin(idx1.chromstr, usechr)] != idx2.start[np.isin(idx2.chromstr, usechr)]):
        return False
    if np.any(idx1.end[np.isin(idx1.chromstr, usechr)] != idx2.end[np.isin(idx2.chromstr, usechr)]):
        return False
    
    return True


def smooth(x: np.array, chromstr: np.array, k: int, x_err: np.array = None):
    """ Smooth a signal chromosome by chromosome.
    
    It simply performs a moving average of size k:
        x_smooth[i] = (x[i-k//2] + ... + x[i+k//2]) / k
    
    If the error array is provided, it assumes that x_i and x_j are independent,
    so the error is simply propagated as:
        x_smooth_err[i] = sqrt(x_err[i-k//2]^2 + ... + x_err[i+k//2]^2) / k

    Args:
        x (np.array): array to smooth
        chromstr (np.array): chromosome array of x
        k (int): window size of the smoothing kernel

    Returns:
        If x_err is None:
            x_smooth (np.array): smoothed array
        If x_err is not None:
            x_smooth (np.array): smoothed array
            x_smooth_err (np.array): smoothed error array
    """
    
    # Initialize the smoothed array
    x_smooth = np.copy(x)
    
    # Initialize the error on the smoothed array, if x_err is provided
    if x_err is not None:
        x_smooth_err = np.copy(x_err)
    
    # Loop over chromosomes, so we don't mix up the signals
    for chrom in np.unique(chromstr):
        mask = chromstr == chrom
        
        # Define the kernel, which is a uniform filter of size k
        kernel = np.ones(k) / k
        
        # Smooth the signal
        # Note: np.convolve with a uniform kernel and mode 'same'
        # is equivalent to a moving average of size k
        x_smooth[mask] = np.convolve(x[mask], kernel, mode='same')
        
        # If the error array is provided, get the error on the smoothed array
        if x_err is not None:
            x_smooth_err[mask] = np.sqrt(np.convolve(x_err[mask]**2, kernel**2, mode='same'))
    
    # If no error array is provided, return the smoothed array
    if x_err is None:
        return x_smooth
    
    # Otherwise, return the smoothed array and the error array
    return x_smooth, x_smooth_err

def clean_correlation(x: np.array, y: np.array,  method: str = 'pearson', return_p: bool = False) -> float:
    """ Pearson or Spearman correlation coefficient, ignoring NaNs and Infs.

    Args:
        x (np.array(n), dtype=float): first input array.
        y (np.array(n), dtype=float): second input array.
        method (str, optional): method to compute the correlation coefficient.
                Either 'pearson' or 'spearman'. Defaults to 'pearson'.
        return_p (bool, optional): if True, return the p-value. Defaults to False.
    
    Returns:
        r (float): Pearson correlation coefficient.
        p (float, optional): p-value (if return_p=True).
    """
    
    # Convert Infs to NaNs
    x[np.isinf(x)] = np.nan
    y[np.isinf(y)] = np.nan
    
    # Remove NaNs (from both arrays)
    mask = np.logical_and(~np.isnan(x), ~np.isnan(y))
    x = x[mask]
    y = y[mask]
    
    # Compute the correlation coefficient
    if method == 'pearson':
        r, p = pearsonr(x, y)
    elif method == 'spearman':
        r, p = spearmanr(x, y)
    else:
        raise ValueError('Method must be either "pearson" or "spearman"')
    
    # Return either just r, or r and p
    if return_p:
        return r, p
    return r


def convert_to_abs_path(cfg: dict):
  """ Given a dictionary or arbitrary depth, convert all relative paths to absolute paths.
  
  This is a recursive function: for each key-value pair, if the value is a file path, it is converted to an absolute path.
  Otherwise, if the value is a dictionary, the function is called recursively on the value.

  Args:
    cfg (dict): Dictionary of arbitrary depth.
  """
  for key, value in cfg.items():
    # If the value is a dictionary, call the function recursively
    if isinstance(value, dict):
      convert_to_abs_path(value)
    # If the value is a file path, convert it to an absolute path
    elif isinstance(value, str) and os.path.exists(value):
      cfg[key] = os.path.abspath(value)


def resample_array(n: int, arr: np.ndarray) -> np.ndarray:
    """ Given an input array, generates a resampled and shuffled
    array of size n.
    
    n can either be equal, less than or more than the length of the input array:
        - if n is equal to the length of arr, the output array is a shuffled version of arr,
        - if n is less than the length of arr, the output array is sampled from the array
            without replacement,
        - if n is more than the length of arr, the output array is created by first tiling
            the input array as much as possible, and then randomly selecting the rest without
            replacement. This ensures that each different element is used as much as possible.

    Args:
        n (int): Size of the resampled array.
        arr (np.ndarray): The array to sample from.

    Returns:
        np.ndarray: Resampled and shuffled array of size n.
    """
    
    # Calculate the number of repetitions and the remainder: n = a * len(idx) + b
    # For example, if n = 430 and len(arr) = 200, then a = 2 and b = 30
    a = n // len(arr)
    b = n % len(arr)

    # Create the repeated part and the remainder part
    arr_out_a = np.tile(arr, a)
    arr_out_b = np.random.choice(arr, b, replace=False)

    # Concatenate the repeated and remainder parts
    arr_out = np.concatenate([arr_out_a, arr_out_b])

    # Shuffle the final array to ensure randomness
    np.random.shuffle(arr_out)

    return arr_out
