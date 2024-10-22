import numpy as np
from alabtools.utils import Index

def impute_cte_trace_data(trace_data: dict, index: Index, chrom: str) -> dict:
    """ Impute 3D coordinates of missing spot data in the trace data dictionary
    for the ChromatinTracingExperiment (CTE) data structure.
    
    The input trace data dictionary should have the form:
    {'spotID1': {
        'x': x, 'y': y, 'z': z, 'chrom': chrom, 'start': start, 'end': end, 'lum': lum },
        ...
    }
    
    The function returns a new trace data dictionary with the same format,
    where:
    - missing spot IDs are called 'IMPUTED_n' (n = 1, 2, ...)
    - the x, y, z coordinates of the missing spots are imputed from the imaged ones,
    - the luminosity of the missing spots is set to NaN.
    
    The code applies a simple 3D linear interpolation:
    - If the missing spot is either at the beginning or the end of the chromosome,
        the spot data is assigned to the closest imaged spot.
    - If the missing spot is between two imaged spots, the spot data is interpolated
        along the line connecting the two imaged spots,
        taking into account the relative genomic positions of the spots.

    Args:
        trace_data (dict): Dictionary of spot data in the format described above.
        index (Index)
        chrom (str): Chromosome of the trace data.

    Returns:
        dict: A dictionary of spot data with the same format as the input trace data,
            where missing spot IDs are imputed. Same format as the input trace data.
    """
    
    # Get the index domain hashmap
    # It is a dictionary mapping the domains to their position in the Index,
    # e.g. {('chr1', 0, 25000): [0, 10000], ...} (for a diploid Index at 25kb)
    index_hash = index.get_index_hashmap()
    
    # Index the trace data by the domain position
    # (with this strategy we can avoid double looping)
    trace_data_indexed = index_trace_data(trace_data, index_hash)
    
    # Create an array with the imaged domain positions
    imgd_domains = np.array([j for j in trace_data_indexed])
    
    # Initialize the imputed trace data dictionary
    trace_data_imp = {}
    # Initialize a counter for the imputed spot IDs
    nspot_imp = 0
    
    # Loop through the index positions
    for i in range(len(index)):
        
        # Skip chromatin domains that are not on the specified chromosome
        if index.chromstr[i] != chrom:
            continue
        
        # If the domain is in the trace data, we don't need to impute
        if i in trace_data_indexed:
            continue
        
        # Otherwise, we need to impute the spot data
        # There are three cases:
        #  1. The domain is at the beginning of the chromosome,
        #     i.e. there are no imaged domains to the left
        #  2. The domain is at the end of the chromosome,
        #     i.e. there are no imaged domains to the right
        #  3. The domain is between two imaged domains
        # In the first two cases we simply assign the spot data
        # as the data of the closest imaged spot,
        # In the third case we impute the spot data from the neighboring imaged ones
        
        # Get the current domain
        chrom = index.chromstr[i]
        start, end = index.start[i], index.end[i]
        
        # Get the neighbors of the domain
        left, right = find_neighbors(i, imgd_domains, trace_data_indexed)
        
        # If both neighbors are None, something went wrong. Raise an error
        if left is None and right is None:
            raise ValueError("Error: no neighbors found for domain")
        
        # If left is None, assign the spot data as the one of the right neighbor
        if left is None:
            x_imp, y_imp, z_imp = right['x'], right['y'], right['z']
        # If right is None, assign the spot data as the one of the left neighbor
        elif right is None:
            x_imp, y_imp, z_imp = left['x'], left['y'], left['z']
        # Otherwise, interpolate the spot data between the two neighbors
        else:
            x_imp, y_imp, z_imp = linear_interpolation(
                start,
                left['x'], left['y'], left['z'], left['start'],
                right['x'], right['y'], right['z'], right['start'],
            )
        
        # Add the imputed spot data to the trace data dictionary
        spotID = f'IMPUTED_{nspot_imp + 1}'
        trace_data_imp[spotID] = {
            'x': x_imp, 'y': y_imp, 'z': z_imp,
            'chrom': chrom, 'start': start, 'end': end,
            'lum': np.nan
        }
        # Increment the spot ID counter
        nspot_imp += 1
    
    # Combine the original trace data with the imputed trace data
    trace_data.update(trace_data_imp)
    
    return trace_data
    

def index_trace_data(trace_data: dict, index_hash: dict) -> dict:
    """ Create a new trace data dictionary indexed by the domain position in the Index.
    
    The output dictionary has the form:
    {i: {'spotID': spotID, 'x': x, 'y': y, 'z': z, 'chrom': chr, 'start': start, 'end': end},
     ...}
    where i is the index position of the domain (chrom, start, end) in the Index from the hashmap.
    (Luminosity is not included in the new dictionary, as it is not needed for interpolation.)

    Args:
        trace_data (dict): Dictionary of spot data with the original format.
        index_hash (dict): Hashmap of the Index domains to their position in the Index.

    Returns:
        dict: Dictionary of spot data indexed by the domain position in the Index.
    """
    
    # Initialize the new dictionary
    trace_data_indexed = {}
    
    # Loop through the spot data
    for spotID in trace_data:
        
        # Get the spot data
        spot_data = trace_data[spotID]
        x, y, z = spot_data['x'], spot_data['y'], spot_data['z']
        chrom, start, end = spot_data['chrom'], spot_data['start'], spot_data['end']
        
        # Get the position of the spot in the Index array using the hash map
        i_domain = index_hash[(chrom, start, end)]
        assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
        i_domain = i_domain[0]
        
        # Add the spot data to the new dictionary
        trace_data_indexed[i_domain] = {
            'spotID': spotID,
            'x': x, 'y': y, 'z': z,
            'chrom': chrom, 'start': start, 'end': end
        }
    
    return trace_data_indexed

def find_neighbors(i: int, imgd_domains: np.ndarray, trace_data_indexed: dict) -> tuple:
    """ Find the closest imaged domains to the left and to the right of the given domain position.
    
    Both left and right are dictionaries with the spot data in the format:
            {'spotID': spotID, 'x': x, 'y': y, 'z': z, 'chrom': chr, 'start': start, 'end': end}
    
    If either left or right neighbors are not found, they are set to None.

    Args:
        i (int): Position of the domain in the Index for which to find neighbors.
        imgd_domains (np.ndarray): Array of the imaged domain positions. shape = (n_imaged_domains,)
        trace_data_indexed (dict): Dictionary of spot data indexed by the domain position in the Index.

    Returns:
        tuple: The left and right neighbors of the domain.
    """

    # Initialize the neighbors to None
    left = None
    right = None
    
    # Calculate the differences between the imaged domain positions and the current domain
    diffs = imgd_domains - i
    
    # Split the differences into those to the left of i and those to the right of i
    mask = diffs < 0
    lefts = diffs[mask]
    rights = diffs[~mask]
    
    # If there are imaged domains to the left, get the closest one
    if len(lefts) > 0:
        l = lefts.max()
        left = trace_data_indexed[l + i]
    
    # If there are imaged domains to the right, get the closest one
    if len(rights) > 0:
        r = rights.min()
        right = trace_data_indexed[r + i]
    
    return left, right
    

def linear_interpolation(
    t: float,
    x1: float, y1: float, z1: float, t1: float,
    x2: float, y2: float, z2: float, t2: float
) -> tuple:
    """ Perform linear interpolation between two 3D points.
    
    Given two points (x1, y1, z1) and (x2, y2, z2) at parameter-values t1 and t2,
    we first find the line connecting the two points in 3D space:
        x(t) = A_x * t + B_x
        y(t) = A_y * t + B_y
        z(t) = A_z * t + B_z
    where A_x = (x2 - x1) / (t2 - t1), B_x = x1 - A_x * t1, etc.

    We then evaluate the line at the parameter-value t to get the interpolated point.
    
    Args:
        t (float): Parameter-value at which to interpolate the point.
        x1 (float): x-coordinate of the first point.
        y1 (float): y-coordinate of the first point.
        z1 (float): z-coordinate of the first point.
        t1 (float): Parameter-value of the first point.
        x2 (float): x-coordinate of the second point.
        y2 (float): y-coordinate of the second point.
        z2 (float): z-coordinate of the second point.
        t2 (float): Parameter-value of the second point.

    Returns:
        tuple: x, y, z coordinates of the interpolated point.
    """
    
    # Rescale t, t1, and t2 to the interval [0, 1]
    # (avoid potential overflow issues if coordinates are very large)
    t = (t - t1) / (t2 - t1)
    t1 = 0
    t2 = 1
    
    # Get the 3D line between the two points,
    # i.e. the line parametrized by t:
    #  x(t) = A_x * t + B_x
    #  y(t) = A_y * t + B_y
    #  z(t) = A_z * t + B_z
    A_x = (x2 - x1) / (t2 - t1)
    B_x = x1 - A_x * t1
    A_y = (y2 - y1) / (t2 - t1)
    B_y = y1 - A_y * t1
    A_z = (z2 - z1) / (t2 - t1)
    B_z = z1 - A_z * t1
    
    # Evaluate the line at t
    x = A_x * t + B_x
    y = A_y * t + B_y
    z = A_z * t + B_z
    
    return x, y, z
