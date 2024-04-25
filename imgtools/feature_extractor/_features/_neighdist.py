import numpy as np
from scipy.spatial.distance import cdist
from ...cte import ChromatinTracingExperiment
from ...cte import cte_utils

docstring = """Measures the 3D distance between a spot and its next neighbor spot along the chromosomal trace up to a certain window size.
If no neighbor is found within the window size, the feature value is set to NaN."""

required_keys = {
    'window_size': {'type': int, 'positive': True},
}

def run(cellID: str, cte: ChromatinTracingExperiment, config: dict, feat_arr: np.ndarray, _) -> np.ndarray:
    """ Calculate the distance to the next neighbor spot (along the chromosomal trace) up to a certain window size.
    
    If no spot is found within the window size, the feature value is set to NaN.
    
    If there are multiple next neighbors within the window size, the median distance is taken.
    
    If there are two or more spots corresponding to the same domain in the trace, the median distance is taken.

    Args:
        cellID (str)
        cte (ChromatinTracingExperiment)
        config (dict)
        feat_arr (np.ndarray): initialized 0-valued array of shape (n_domains, n_traces) to store the feature values
        _: not used, just to match the function signature

    Returns:
        (np.ndarray): updated array of shape (n_domains, n_traces) with the feature values
    """
    
    # Get the window size from the config
    try:
        window_size = config['window_size']
    except KeyError:
        raise KeyError("Error: window_size not found in the config for the feature 'neighdist'")
    
    # Get the cell data in dictionary format
    cell_data = cte.get_data(cellID)
    
    # Get the traceID hash table to map traces to their position in the array
    traceID_hash = cte.get_trace_hashmap(cellID)
    
    # Get the index and its hash table
    index = cte.index
    index_hash = index.get_index_hashmap()
    
    # Initialize a dictionary to store the feature values for each domain (we will then take the median)
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
                
                # Get the position of the spot in the Index array using the hash tables
                i_domain = index_hash[(chrom, start, end)]
                assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
                i_domain = i_domain[0]
                
                # Initialize the list of values for this domain if necessary
                if (i_domain, i_trace) not in feat_per_domain:
                    feat_per_domain[(i_domain, i_trace)] = []
                
                # Select the spots whose start positions are within the window (after the start of the current spot)
                mask_win = np.logical_and(starts - start >= 0, starts - start <= window_size)
                crds_win = crds[mask_win, :]
                
                # If there is only one spot in the window, set the feature value to NaN
                if crds_win.shape[0] == 1:
                    feat_per_domain[(i_domain, i_trace)].append(np.nan)
                    continue
                
                # Get the distance between each spot of the window and the spot of interest
                dists = cdist(np.array([[x, y, z]]), crds_win).flatten()
                # Remove the self-distance
                dists = dists[dists > 0]
                
                # Compute the median distance
                med_dist = np.median(dists)
                
                # Add the feature value to the dictionary of values for this domain
                feat_per_domain[(i_domain, i_trace)].append(med_dist)
                
                del mask_win, crds_win, dists, med_dist
                
    
    # Compute the median of the values for each domain and add them to the feature array
    for (i_domain, i_trace), vals in feat_per_domain.items():
        feat_arr[i_domain, i_trace] = np.nanmedian(vals)
    
    return feat_arr
