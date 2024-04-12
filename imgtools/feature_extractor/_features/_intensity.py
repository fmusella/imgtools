import numpy as np
from ...cte import ChromatinTracingExperiment

required_keys = {
    'method': {'type': str},
}

AVAILABLE_METHODS = ['median', 'sum']

def run(cellID: str, cte: ChromatinTracingExperiment, config: dict, feat_arr: np.ndarray, _) -> np.ndarray:
    """ Calculate the intensity of each domain.
    
    If there are two or more spots corresponding to the same domain in the trace, the median intensity is taken.

    Args:
        cellID (str)
        cte (ChromatinTracingExperiment)
        config (dict)
        feat_arr (np.ndarray): initialized 0-valued array of shape (n_domains, n_traces) to store the intensity values
        _: not used, just to match the function signature

    Returns:
        (np.ndarray): updated array of shape (n_domains, n_traces) with the intensity values
    """
    
    # Get the method from the config
    try:
        method = config['method']
    except KeyError:
        raise KeyError("Error: method not specified in the config for intensity feature")
    # Check if the method is valid
    if method not in AVAILABLE_METHODS:
        raise ValueError(f"Error: method {method} not available for intensity feature")
    
    # Get the cell data in dictionary format
    cell_data = cte.get_data(cellID)
    
    # Get the index and its hash table
    index = cte.index
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
            
            for spotID in cell_data[chrom][traceID]:
                
                # Unpack the spot data
                spot_data = cell_data[chrom][traceID][spotID]
                lum = spot_data['lum']
                start, end = spot_data['start'], spot_data['end']
                
                # Get the position of the spot in the Index array using the hash tables
                i_domain = index_hash[(chrom, start, end)]
                assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
                i_domain = i_domain[0]
                
                # Add the feature value to the dictionary of values for this domain (initialize if necessary)
                if (i_domain, i_trace) not in feat_per_domain:
                    feat_per_domain[(i_domain, i_trace)] = []
                feat_per_domain[(i_domain, i_trace)].append(lum)
                

    # Compute the ensemble of the values (specified by the method) for each domain and add them to the feature array
    for (i_domain, i_trace), vals in feat_per_domain.items():
        if method == 'median':
            feat_arr[i_domain, i_trace] = np.median(vals)
        elif method == 'sum':
            feat_arr[i_domain, i_trace] = np.sum(vals)
    
    return feat_arr
