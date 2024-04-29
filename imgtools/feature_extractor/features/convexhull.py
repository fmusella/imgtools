import numpy as np
from scipy.spatial import ConvexHull
from ...cte import ChromatinTracingExperiment
from ...cte import cte_utils

docstring = """Calculates the volume of the Convex Hull fitted on the spots that are within a genomic window centered at each spot.
The Convex Hull of a set of points is the smallest convex polygon that contains all of them.
In 3D, at least 4 non-coplanar points are needed to define a Convex Hull. If this condition is not met, the volume is set to NaN."""

required_keys = {
    'window_size': {'type': int, 'positive': True},
}

def run(cellID: str, cte: ChromatinTracingExperiment, config: dict, feat_arr: np.ndarray, _) -> np.ndarray:
    """ Calculates the volume of the Convex Hull fitted on the spots that are within a genomic window centered at each spot.
    
    The Convex Hull of a set of points is the smallest convex polygon that contains all of them.
    
    In 3D, at least 4 non-coplanar points are needed to define a Convex Hull. If this condition is not met, the volume is set to NaN.
    
    If there are two or more spots corresponding to the same domain in the trace, the average value is taken.

    Args:
        cellID (str)
        cte (ChromatinTracingExperiment)
        config (dict): configuration dictionary with the following keys:
            - window_size (int): size of the genomic window in bp
        feat_arr (np.ndarray): initialized nan-valued array of shape (n_domains, n_traces) to store the feature values
        _: not used, just to match the signature of the function

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
                mask_win = np.abs(starts - start) <= window_size / 2
                crds_win = crds[mask_win, :]
                
                # Try to fit a Convex Hull to the points
                try:
                    hull = ConvexHull(crds_win)
                except:
                    # If the Convex Hull cannot be calculated, skip this spot (the feature value is kept as NaN)
                    continue

                # Calculate the volume of the Convex Hull
                feat_val = hull.volume
                
                # If the volume is NaN or infinite, skip this spot (the feature value is kept as NaN)
                if np.isnan(feat_val) or np.isinf(feat_val):
                    continue
                
                # Get the position of the spot in the Index array using the hash tables
                i_domain = index_hash[(chrom, start, end)]
                assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
                i_domain = i_domain[0]
                
                # Initialize the list of values for this domain if necessary
                if (i_domain, i_trace) not in feat_per_domain:
                    feat_per_domain[(i_domain, i_trace)] = []
                
                # Add the feature value to the dictionary of values for this domain
                feat_per_domain[(i_domain, i_trace)].append(feat_val)
                
    
    # Compute the average of the values for each domain and add them to the feature array
    for (i_domain, i_trace), vals in feat_per_domain.items():
        feat_arr[i_domain, i_trace] = np.nanmean(vals)
    
    return feat_arr
