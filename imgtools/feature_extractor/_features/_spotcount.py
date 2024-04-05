import numpy as np
from alabtools.utils import Index

def run(count_arr: np.ndarray, cell_data: dict, index: Index) -> tuple:
    """ Counts the number of spots per domain and per trace in the cell.

    Args:
        count_arr (np.ndarray): initialized 0-valued array of shape (ndomain, max_ntrace_per_chrom) to store the number of spots
        cell_data (dict): Data of the cell in dictionary format
        index (Index)

    Returns:
        np.ndarray: Updated array of shape (n_domains, n_traces) with the number of spots
    """
    
    # Convert the  count_arr to an array of 0s
    count_arr = np.zeros(count_arr.shape, dtype=count_arr.dtype)
    
    # Create a hash table for the index
    index_hash = index.get_index_hashmap()
    
    for chrom in cell_data:
            
        # Get the traces in the chromosome and hash them
        traceIDs = list(cell_data[chrom].keys())
        traceIDs.sort()  # Sort to ensure that the order doesn't depend on how the dictionary is iterated
        traceID_hash = {traceID: i for i, traceID in enumerate(traceIDs)}
        
        for traceID in cell_data[chrom]:
            
            # Get the position of the trace in the array using the hash tables
            i_trace = traceID_hash[traceID]
            
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
                count_arr[i_domain, i_trace] += 1
    
    return count_arr
