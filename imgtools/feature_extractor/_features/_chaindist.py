import numpy as np
from scipy.spatial.distance import cdist
from ...cte import ChromatinTracingExperiment
from ...cte import cte_utils

docstring = """Measures the average 3D distance between a spot and its chain neighbors (i-1 and i+1).
If the chain neighbors are not present, the feature value is set to NaN.
This feature requires a regular resolution in the Index."""

required_keys = {}

def run(cellID: str, cte: ChromatinTracingExperiment, _1, feat_arr: np.ndarray, _2) -> np.ndarray:
    """ Calculate the average 3D distance between a spot and its chain neighbors (i-1 and i+1).
    
    If the chain neighbors are not present, the feature value is set to NaN.
    
    This feature requires a regular resolution in the Index.
    
    If there are two or more spots corresponding to the same domain in the trace, the average distance is taken.

    Args:
        cellID (str)
        cte (ChromatinTracingExperiment)
        feat_arr (np.ndarray): initialized 0-valued array of shape (n_domains, n_traces) to store the feature values
        _*: not used, just to match the function signature

    Returns:
        (np.ndarray): updated array of shape (n_domains, n_traces) with the feature values
    """

    # Get the cell data in dictionary format
    cell_data = cte.get_data(cellID)
    
    # Get the traceID hash table to map traces to their position in the array
    traceID_hash = cte.get_trace_hashmap(cellID)
    
    # Get the index, its hash table and the resolution
    index = cte.index
    index_hash = index.get_index_hashmap()
    res = index.resolution()
    
    # If the resolution is not regular raise an error
    if res is None:
        raise ValueError("The Index resolution is not regular. Chain distance calculation is not possible.")
    
    # Initialize a dictionary to store the feature values for each domain (we will then take the average)
    feat_per_domain = {}
    
    for chrom in cell_data:        
        for traceID in cell_data[chrom]:
            
            # Get the position of the trace in the array
            i_trace = traceID_hash[chrom][traceID]
            
            # Get the trace data in numpy format
            trace_data = cell_data[chrom][traceID]
            xs, ys, zs, starts, _, _, _ = cte_utils.trace_dict_to_numpy(trace_data)
            crds = np.array([xs, ys, zs]).T
            
            for spotID in cell_data[chrom][traceID]:
                
                # Unpack the spot data
                spot_data = cell_data[chrom][traceID][spotID]
                start, end = spot_data['start'], spot_data['end']
                x, y, z = spot_data['x'], spot_data['y'], spot_data['z']
                
                # Select the spots whose start position is equal to +/- one resolution unit from the spot of interest
                chain_mask = np.abs(starts - start) == res
                crds_chain = crds[chain_mask, :]
                
                # If there are no chain neighbors, skip this spot (the feature value will stay NaN)
                if crds_chain.shape[0] == 0:
                    continue
                
                # Get the distance between the spot and its chain neighbors
                dists = cdist(np.array([[x, y, z]]), crds_chain).flatten()
                
                # Get the position of the spot in the Index array using the hash tables
                i_domain = index_hash[(chrom, start, end)]
                assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
                i_domain = i_domain[0]
                
                # Initialize the list of values for this domain if necessary
                if (i_domain, i_trace) not in feat_per_domain:
                    feat_per_domain[(i_domain, i_trace)] = []
                
                # Add the average distance to the list of values for this domain
                feat_per_domain[(i_domain, i_trace)].append(np.mean(dists))
                
    
    # Compute the average of the values for each domain and add them to the feature array
    for (i_domain, i_trace), vals in feat_per_domain.items():
        feat_arr[i_domain, i_trace] = np.nanmean(vals)
    
    return feat_arr
