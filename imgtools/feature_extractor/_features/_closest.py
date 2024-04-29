import numpy as np
from scipy.spatial.distance import cdist
from ...cte import ChromatinTracingExperiment
from ...cte import cte_utils

docstring = """Measures the average distance between a spot and its N closest 3D neighbors."""

required_keys = {
    'nclosest': {'type': int, 'positive': True},
}

def run(cellID: str, cte: ChromatinTracingExperiment, config: dict, feat_arr: np.ndarray, _) -> np.ndarray:
    """ Measures the average distance between a spot and its N closest 3D neighbors.
    
    If two or more spots are mapped to the same domain, the average of the values is taken.

    Args:
        cellID (str)
        cte (ChromatinTracingExperiment)
        config (dict)
        feat_arr (np.ndarray): initialized 0-valued array of shape (n_domains, n_traces) to store the feature value.
        _: not used, just to match the function signature

    Returns:
        (np.ndarray): updated array of shape (n_domains, n_traces) with the feature values.
    """
    
    # Get the number of closest neighbors to consider
    nclosest = config['nclosest']
    
    # Get the cell data in dictionary format
    cell_data = cte.get_data(cellID)
    
    # Get the traceID hash table to map traces to their position in the array
    traceID_hash = cte.get_trace_hashmap(cellID)
    
    # Convert the cell data in numpy format and get the coordinates of each spot
    xs, ys, zs, _, _, _, _, _, _ = cte_utils.cell_dict_to_numpy(cell_data)
    crds = np.array([xs, ys, zs]).T
    
    # Get the index and its hash table
    index = cte.index
    index_hash = index.get_index_hashmap()
    
    # Initialize a dictionary to store the feature values for each domain (we will then take the average)
    feat_per_domain = {}
    
    for chrom in cell_data:       
        for traceID in cell_data[chrom]:
            
            # Get the position of the trace in the array
            i_trace = traceID_hash[chrom][traceID]
            
            for spotID in cell_data[chrom][traceID]:
                
                # Unpack the spot data
                spot_data = cell_data[chrom][traceID][spotID]
                x, y, z = spot_data['x'], spot_data['y'], spot_data['z']
                start, end = spot_data['start'], spot_data['end']

                # Calculate the distance of this spot to all the other spots
                point = np.array([[x, y, z]])
                dists = cdist(point, crds).flatten()
                
                # Remove the distance to itself
                dists = dists[dists != 0]
                
                # Use numpy.partition to get the smallest N distances
                # numpy.partition(x, k) creates a new array where the k-th element is the position it would have in a sorted array,
                # all elements to the left are smaller than it, and all elements to the right are larger.
                nclosest_dists = np.partition(dists, nclosest-1)[:nclosest]
                
                # Compute the feature value as the average of the N closest distances
                feat_val = np.mean(nclosest_dists)
                
                # Get the position of the spot in the Index array using the hash tables
                i_domain = index_hash[(chrom, start, end)]
                assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
                i_domain = i_domain[0]
                
                # Initialize the list of values for this domain if necessary
                if (i_domain, i_trace) not in feat_per_domain:
                    feat_per_domain[(i_domain, i_trace)] = []
                
                # Add the feature value to the dictionary of values for this domain (initialize if necessary)
                feat_per_domain[(i_domain, i_trace)].append(feat_val)
                
    
    # Compute the average of the values for each domain and add them to the feature array
    for (i_domain, i_trace), vals in feat_per_domain.items():
        feat_arr[i_domain, i_trace] = np.nanmean(vals)
    
    return feat_arr
