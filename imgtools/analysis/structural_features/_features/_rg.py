import numpy as np
from scipy.spatial.distance import cdist
from alabtools.utils import Index

required_keys = {
    'window_size': {'type': int, 'positive': True},
}

def run(feat_arr: np.ndarray, cell_data: dict, index: Index, config: dict) -> tuple:
    """ Calculate the radius of gyration for each spot.
    
    The radius of gyration is computed, for each spot, withing a genomic window (specified in the config).
    
    The radius of gyration for spot i is defined as:
        gyr_i = sqrt( (1/N) * sum_j( (r_ij)^2 ) ),
    where
        - the sum_j is over all the spots j within the genomic window centered at i,
        - N is the number of spots in the genomic window,
        - r_ij is the distance between spot j and the center of mass of all the spots in the genomic window.
    
    The general formula also includes the mass of the spots, but we are assuming that all the spots have the same mass.
    
    If there are two or more spots corresponding to the same domain in the trace, the median radius of gyration is taken.

    Args:
        feat_arr (np.ndarray): initialized 0-valued array of shape (n_domains, n_traces) to store the radius of gyrations
        cell_data (dict): data of the cell in dictionary format
        index (Index)
        config (dict): configuration dictionary

    Returns:
        (np.ndarray): updated array of shape (n_domains, n_traces) with the radii of gyrations
    """
    
    # Get the window size from the config
    try:
        window_size = config['window_size']
    except KeyError:
        raise KeyError("Error: window_size not found in the config for the radius of gyration feature.")
    
    # Check that the index has a finite resolution
    index_res = index.resolution()
    if index_res is None:
        raise ValueError("Error: the index does not have a finite resolution.")
    
    # If the window size is too small, raise an error
    if window_size < 3 * index_res:
        raise ValueError("Error: the window size is too small. It should be at least 3 times the index resolution.")
    
    # Get the window size in units of the index resolution
    # Convert the window size to the smallest odd integer greater than or equal than it
    window_size = int(np.ceil(window_size / index_res))
    if window_size % 2 == 0:
        window_size += 1
    
    # Loop through the index and assing each domain to the windows it belongs to
    # We include this in the domain_to_windows dictionary, whose structure is:
    #       domain_to_windows[(chrom, start, end)] = [
    #           (chrom_1, start_1, end_1),
    #           (chrom_2, start_2, end_2),
    #           ...]
    # where the list contains the domains that are contained in the window,
    # and vice-verse, the domain is assigned to the windows centered at the domains in the list
    domain_to_windows = {}
    for i, (chrom, start, end) in enumerate(zip(index.chromstr, index.start, index.end)):
        # Get the lower and upper bounds of the window
        low = int(i - (window_size - 1) / 2)
        high = int(i + (window_size - 1) / 2)
        # If the window is out of bounds, adjust the bounds (i.e. increase low or decrease high)
        # This means that the window is going to be smaller and not centered around i
        # First, adjust the bounds if they are out of the index
        if low < 0:
            low = 0
        if high >= len(index):
            high = len(index) - 1
        # Second, adjust the bounds if they are out of the chromosome
        while index.chromstr[low] != chrom:
            low += 1
        while index.chromstr[high] != chrom:
            high -= 1
        # Now create the list of domains that are contained in the window
        windows = []
        for j in range(low, high + 1):
            chrom_j, start_j, end_j = index.chromstr[j], index.start[j], index.end[j]
            windows.append((chrom_j, start_j, end_j))
        domain_to_windows[(chrom, start, end)] = windows
    
    # Loop through the cell data and assign the spots to the windows they belong to
    # We store this in the windows_spots dictionary, whose structure is:
    #       windows_spots[chrom][traceID][(chrom, start, end)] = [
    #           [x_1, y_1, z_1],
    #           [x_2, y_2, z_2],
    #           ...]
    # where the list contains the spots that are contained in the window.
    windows_spots = {}
    for chrom in cell_data:
        for traceID in cell_data[chrom]:
            for spotID in cell_data[chrom][traceID]:
                
                # Unpack the spot data
                spot_data = cell_data[chrom][traceID][spotID]
                x, y, z = spot_data['x'], spot_data['y'], spot_data['z']
                start, end = spot_data['start'], spot_data['end']
                
                # Get the windows the spot belongs to
                spot_windows = domain_to_windows((chrom, start, end))
                
                # Add the spot to all the windows it belongs to, creating the keys if necessary
                if chrom not in windows_spots:  # add the chromosome if not present
                    windows_spots[chrom] = {}
                if traceID not in windows_spots[chrom]:  # add the traceID if not present
                    windows_spots[chrom][traceID] = {}
                for window in spot_windows:
                    if window not in windows_spots[chrom][traceID]:  # add the window if not present
                        windows_spots[chrom][traceID][window] = []
                    # Add the spot to the window, appending the [x, y, z] coordinates (as a list of 3 elements)
                    windows_spots[chrom][traceID][window].append([x, y, z])
    
    # Get the hash table for the index
    index_hash = index.get_index_hashmap()
    
    # Loop through the windows and compute the radius of gyration for each spot
    for chrom in windows_spots:
        
        # Get the traces in the chromosome and hash them
        traceIDs = list(cell_data[chrom].keys())
        traceIDs.sort()  # Sort to ensure that the order doesn't depend on how the dictionary is iterated
        traceID_hash = {traceID: i for i, traceID in enumerate(traceIDs)}
        
        for traceID in windows_spots[chrom]:
            
            # Get the position of the trace in the Index array
            i_trace = traceID_hash[traceID]
            
            for window in windows_spots[chrom][traceID]:
                
                # Unpack the window domain
                chrom, start, end = window
                
                # Get the position of the spot in the Index array using the hash tables
                i_domain = index_hash[(chrom, start, end)]
                assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
                i_domain = i_domain[0]
                
                # Get the spots in the window
                spots = windows_spots[chrom][traceID][window]
                spots = np.array(spots)
                
                # Get the center of mass of the spots
                center_of_mass = np.mean(spots, axis=0)
                
                # Compute the distance between each spot and the center of mass
                dists = cdist(spots, center_of_mass[None, :])
                
                # Compute the radius of gyration
                gyr = np.sqrt(np.mean(dists**2))
                
                # Add the radius of gyration to the feature array
                feat_arr[i_domain, i_trace] = gyr
    
    return feat_arr
