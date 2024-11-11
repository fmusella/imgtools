import numpy as np
from ...cte import ChromatinTracingExperiment

docstring = """Gets the luminescence intensity of each spot.
When multiple spots correspond to the same domain in a trace, their average is taken."""

required_keys = {
    'method': {'type': str},
}

def run(cellID: str, cte: ChromatinTracingExperiment, config: dict, feat_arr: np.ndarray, _) -> np.ndarray:
    """ Gets the intensity of each domain.
    
    If there are two or more spots corresponding to the same domain in the trace:
        - if method is 'mean', the average intensity is taken
        - if method is 'sum', the sum of the intensities is taken

    Args:
        cellID (str)
        cte (ChromatinTracingExperiment)
        config (dict): configuration dictionary with the following keys:
            - method (str): either 'mean' or 'sum'
        feat_arr (np.ndarray): initialized nan-valued array of shape (n_domains, n_traces) to store the feature values
        _: not used, just to match the function signature

    Returns:
        (np.ndarray): updated array of shape (n_domains, n_traces) with the feature values
    """
    
    # Check that the 'method' key is either 'mean' or 'sum'
    if config['method'] not in ['mean', 'sum']:
        raise ValueError(f"Error: 'method' must be either 'mean' or 'sum'. Got {config['method']}")
    
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
            
            for spotID in cell_data[chrom][traceID]:
                
                # Unpack the spot data
                spot_data = cell_data[chrom][traceID][spotID]
                lum = spot_data['lum']
                start, end = spot_data['start'], spot_data['end']
                
                # Get the position of the spot in the Index array using the hash tables
                i_domain = index_hash[(chrom, start, end)]
                assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
                i_domain = i_domain[0]
                
                # Initialize the list of values for this domain if necessary
                if (i_domain, i_trace) not in feat_per_domain:
                    feat_per_domain[(i_domain, i_trace)] = []
                
                # Add the feature value to the dictionary of values for this domain
                feat_per_domain[(i_domain, i_trace)].append(lum)
                

    # Compute the average of the values for each domain and add them to the feature array
    for (i_domain, i_trace), vals in feat_per_domain.items():
        if config['method'] == 'mean':
            feat_arr[i_domain, i_trace] = np.nanmean(vals)
        elif config['method'] == 'sum':
            feat_arr[i_domain, i_trace] = np.nansum(vals)
    
    return feat_arr
