import numpy as np
from ...cte import ChromatinTracingExperiment

docstring = """Counts the number of spots per domain, seperately for each trace. Missing data are considered as 0."""

required_keys = {}

def run(cellID: str, cte: ChromatinTracingExperiment, _1, feat_arr: np.ndarray, _2) -> np.ndarray:
    """ Counts the number of spots per domain, seperately for each trace.

    Args:
        cellID (str)
        cte (ChromatinTracingExperiment)
        feat_arr (np.ndarray): feature array of shape (n_domains, n_traces), to be updated with the number of spots
        _*: not used, just to match the function signature
    
    Returns:
        np.ndarray: Updated array of shape (n_domains, n_traces) with the number of spots
    """
    
    # Get the cell data in dictionary format
    cell_data = cte.get_data(cellID)
    
    # Get the traceID hash table to map traces to their position in the array
    traceID_hash = cte.get_trace_hashmap(cellID)
    
    # Convert the feat_arr to an array of 0s
    feat_arr = np.zeros(feat_arr.shape, dtype=feat_arr.dtype)
    
    # Get the index object and get the hash table
    index = cte.index
    index_hash = index.get_index_hashmap()
    
    for chrom in cell_data:        
        for traceID in cell_data[chrom]:
            
            # Get the position of the trace in the array using the hash tables
            i_trace = traceID_hash[chrom][traceID]
            
            for spotID in cell_data[chrom][traceID]:
                
                # Unpack the spot data
                spot_data = cell_data[chrom][traceID][spotID]
                start, end = spot_data['start'], spot_data['end']
                
                # Get the position of the spot in the array using the hash tables
                # index_hash is a dictionary whose keys are tuples (chrom, start, end) and values are lists of indices,
                # where each index corresponds to the position of that domain in the Index arrays for each copy.
                # If the Index is haploid the list has only one element, if the Index is diploid the list has two elements, etc.
                i_domain = index_hash[(chrom, start, end)]
                assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
                i_domain = i_domain[0]
                
                # Increment the count
                feat_arr[i_domain, i_trace] += 1
    
    
    return feat_arr
