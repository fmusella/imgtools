import numpy as np
from scipy.spatial.distance import cdist
from ...cte import ChromatinTracingExperiment
from ...cte import cte_utils

docstring = """For each spot, it measures the radius of gyration within a genomic window centered at the spot. 
The radius of gyration is defined as: gyr_i = sqrt( (1/N) * sum_j( (r_j[i])^2 ) ),
where
    - the sum_j is over all the spots j within the genomic window centered at i,
    - N is the number of spots in the genomic window,
    - r_j[i] is the distance between spot j of the window and the center of mass of all the spots in the genomic window. 
If there are no other spots in the window, the radius of gyration is set to NaN."""

required_keys = {
    'window_size': {'type': int, 'positive': True},
}

def run(cellID: str, cte: ChromatinTracingExperiment, config: dict, feat_arr: np.ndarray, _) -> np.ndarray:
    """ Calculate the radius of gyration for each spot.
    
    The radius of gyration is computed, for each spot, withing a genomic window (specified in the config).
    
    The radius of gyration for spot i is defined as:
        gyr_i = sqrt( (1/N) * sum_j( (r_j[i])^2 ) ),
    where
        - the sum_j is over all the spots j within the genomic window centered at i,
        - N is the number of spots in the genomic window,
        - r_j[i] is the distance between spot j of the window and the center of mass of all the spots in the genomic window.
    
    The general formula also includes the mass of the spots, but we are assuming that all the spots have the same mass.
    
    If there are two or more spots corresponding to the same domain in the trace, the average value is taken.

    Args:
        feat_arr (np.ndarray): initialized 0-valued array of shape (n_domains, n_traces) to store the feature values
        cell_data (dict): data of the cell in dictionary format
        index (Index)
        config (dict): configuration dictionary

    Returns:
        (np.ndarray): updated array of shape (n_domains, n_traces) with the feature values
    """
    
    # Get the window size from the config
    window_size = config['window_size']
    
    # Get the cell data in dictionary format
    cell_data = cte.get_data(cellID)
    
    # Get the traceID hash table to map traces to their position in the array
    traceID_hash = cte.get_trace_hashmap(cellID)
    
    # Get the index and its hash table
    index = cte.index
    index_hash = index.get_index_hashmap()
    
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
                
                # Select the spots whose start positions are within half
                # the window size from the start of the current spot
                mask_win = np.abs(starts - start) < window_size / 2
                crds_win = crds[mask_win, :]
                
                # If there is only one spot in the window, skip this spot (feature value is kept as NaN)
                if crds_win.shape[0] == 1:
                    continue

                # Calculate the center of mass
                com = np.mean(crds_win, axis=0)
                
                # Get the distance between each spot and the center of mass
                dists = cdist(crds_win, com[None, :])
                
                # Calculate the radius of gyration
                gyr = np.sqrt(np.mean(dists**2))
                
                # Get the position of the spot in the Index array using the hash tables
                i_domain = index_hash[(chrom, start, end)]
                assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
                i_domain = i_domain[0]
                
                # Initialize the list of values for this domain if necessary
                if (i_domain, i_trace) not in feat_per_domain:
                    feat_per_domain[(i_domain, i_trace)] = []
                
                # Add the feature value to the dictionary of values for this domain
                feat_per_domain[(i_domain, i_trace)].append(gyr)
                
    
    # Compute the average of the values for each domain and add them to the feature array
    for (i_domain, i_trace), vals in feat_per_domain.items():
        feat_arr[i_domain, i_trace] = np.nanmean(vals)
    
    return feat_arr
