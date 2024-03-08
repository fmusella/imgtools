import numpy as np
import h5py
from alabtools.utils import Index

def run(cellID: str, feature: str, feat_arr: np.ndarray, cell_data: dict, index: Index, config: dict) -> tuple:
    """ Counts the number of spots per domain and per trace in the cell.

    Args:
        count_arr (np.ndarray): initialized 0-valued array of shape (ndomain, max_ntrace_per_chrom) to store the number of spots
        cell_data (dict): Data of the cell in dictionary format
        index (Index)

    Returns:
        np.ndarray: Updated array of shape (n_domains, n_traces) with the number of spots
        None: Not used, just to match the return of the function
    """
    
    # Get the ImmunoFluorescence HDF5 file
    imf_h5 = h5py.File(config['IF_file'], 'r')
    
    # Make sure the feature is in the HDF5 file
    if feature not in imf_h5[cellID]:
        raise ValueError(f"Feature {feature} not found in the HDF5 file for cell {cellID}")
    
    # Get the data - a numpy array of shape (nspot,) - for this feature in the cell
    imf_vals = imf_h5[cellID][feature][:]
    # Get the spotIDs associated with the imf array
    imf_spotIDs = imf_h5[cellID]['spotIDs'][:]
    imf_h5.close()
    
    # Hash the spotIDs with their ImF value
    imf_data = {}
    for spotID, val in zip(imf_spotIDs, imf_vals):
        imf_data[spotID] = val
    
    # Get the hash table for the index
    index_hash = index.get_index_hashmap()

    # Initialize a dictionary to store the feature values for each domain (we will then take the median)
    feat_per_domain_vals = {}
    
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
                i_domain = index_hash[(chrom, start, end)]
                assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
                i_domain = i_domain[0]
                
                # Add the ImF value to the dictionary of values for this domain (initialize if necessary)
                if (i_domain, i_trace) not in feat_per_domain_vals:
                    feat_per_domain_vals[(i_domain, i_trace)] = []
                feat_per_domain_vals[(i_domain, i_trace)].append(imf_data[spotID])
    
    # Compute the median of the values for each domain
    for (i_domain, i_trace), vals in feat_per_domain_vals.items():
        feat_arr[i_domain, i_trace] = np.median(vals)
    
    return feat_arr, None
