import numpy as np
from scipy.spatial.distance import cdist
from alabtools.utils import Index
from ...cte import cte_utils

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
    
    # Get the hash table for the index
    index_hash = index.get_index_hashmap()
    
    # Initialize a dictionary to store the feature values for each domain (we will then take the median)
    feat_per_domain = {}
    
    for chrom in cell_data:
            
        # Get the traces in the chromosome and hash them
        traceIDs = list(cell_data[chrom].keys())
        traceIDs.sort()  # Sort to ensure that the order doesn't depend on how the dictionary is iterated
        traceID_hash = {traceID: i for i, traceID in enumerate(traceIDs)}
        
        for traceID in cell_data[chrom]:
            
            # Get the position of the trace in the array
            i_trace = traceID_hash[traceID]
            
            # Get the trace data in numpy format
            trace_data = cell_data[chrom][traceID]
            xs, ys, zs, starts, _, _, _ = cte_utils.trace_dict_to_numpy(trace_data)
            crds = np.array([xs, ys, zs]).T
            
            for spotID in cell_data[chrom][traceID]:
                
                # Unpack the spot data
                spot_data = cell_data[chrom][traceID][spotID]
                start, end = spot_data['start'], spot_data['end']
                
                # Get the position of the spot in the Index array using the hash tables
                i_domain = index_hash[(chrom, start, end)]
                assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
                i_domain = i_domain[0]
                
                # Initialize the list of values for this domain if necessary
                if (i_domain, i_trace) not in feat_per_domain:
                    feat_per_domain[(i_domain, i_trace)] = []
                
                # Select the spots whose start positions are within half
                # the window size from the start of the current spot
                mask_win = np.abs(starts - start) < window_size / 2
                crds_win = crds[mask_win, :]
                
                # If there is only one spot in the window, set the radius of gyration to NaN
                if crds_win.shape[0] == 1:
                    feat_per_domain[(i_domain, i_trace)].append(np.nan)
                    continue

                # Calculate the center of mass
                com = np.mean(crds_win, axis=0)
                
                # Get the distance between each spot and the center of mass
                dists = cdist(crds_win, com[None, :])
                
                # Calculate the radius of gyration
                gyr = np.sqrt(np.mean(dists**2))
                
                # Add the feature value to the dictionary of values for this domain
                feat_per_domain[(i_domain, i_trace)].append(gyr)
                
                del mask_win, crds_win, com, dists, gyr
                
    
    # Compute the median of the values for each domain and add them to the feature array
    for (i_domain, i_trace), vals in feat_per_domain.items():
        feat_arr[i_domain, i_trace] = np.nanmedian(vals)
    
    return feat_arr
