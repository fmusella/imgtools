import os
import numpy as np
import h5py
from ...cte import ChromatinTracingExperiment

docstring = """Extracts the ImmunoFluorescence (ImF) measurement for each spot in the cell. 
The ImF values have been independently measured and stored in a separate file.
There are many markers that have been measured, so the feature name is used to specify which one is extracted.
For example, the feature name could be 'SF3A66', which is the name of a Speckle-associated marker."""

required_keys = {
    'ImF_file': {'type': str}
}

def run(cellID: str, cte: ChromatinTracingExperiment, config: dict, feat_arr: np.ndarray, feature: str) -> np.ndarray:
    """ Get the ImmunoFluorescence (ImF) values for the spots in the cell and store them in the feature array.
    
    The ImF values are stored in the HDF5 file that is specified in the configuration file.
    
    If multiple spots are associated with the same domain, the average of the ImF values is taken.

    Args:
        cellID (str)
        cte (ChromatinTracingExperiment)
        config (dict)
        feat_arr (np.ndarray): initialized 0-valued array of shape (ndomain, max_ntrace_per_chrom) to store the feature values
        feature (str): Name of the feature to extract

    Returns:
        np.ndarray: Updated array of shape (n_domains, n_traces) with the feature values
    """
    
    # Check that the ImF file exists
    if not os.path.isfile(config['ImF_file']):
        raise ValueError(f"The ImmunoFluorescence file {config['ImF_file']} does not exist.")
    
    # Get the ImmunoFluorescence HDF5 file, raise an error if h5py can't open it
    try:
        imf_h5 = h5py.File(config['ImF_file'], 'r')
    except Exception as e:
        raise ValueError(f"Error opening the ImmunoFluorescence file as HDF5 file: {e}")
    
    # Make sure the feature is in the HDF5 file
    if feature not in imf_h5[cellID]:
        raise ValueError(f"Feature {feature} not found in the HDF5 file for cell {cellID}")
    
    # Get the data - a numpy array of shape (nspot,) - for this feature in the cell
    imf_vals = imf_h5[cellID][feature][:]
    # Get the spotIDs associated with the imf array
    imf_spotIDs = imf_h5[cellID]['spotIDs'][:].astype('U20')
    # Check that the lenghts match
    if len(imf_vals) != len(imf_spotIDs):
        raise ValueError(f"Length mismatch between ImF values and spotIDs for feature {feature}")
    imf_h5.close()
    
    # Hash the spotIDs with their ImF value: imf_data[spotID] = imf_val
    imf_data = {}
    for spotID, val in zip(imf_spotIDs, imf_vals):
        imf_data[spotID] = val
    
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
            
            # Get the position of the trace in the array using the hash tables
            i_trace = traceID_hash[chrom][traceID]
            
            for spotID in cell_data[chrom][traceID]:
                
                # Unpack the spot data
                spot_data = cell_data[chrom][traceID][spotID]
                start, end = spot_data['start'], spot_data['end']
                
                # Get the feature value for this spot
                feat_val = imf_data[spotID]
                
                # Get the position of the spot in the array using the hash tables
                i_domain = index_hash[(chrom, start, end)]
                assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
                i_domain = i_domain[0]
                
                # Initialize the list of values for this domain if necessary
                if (i_domain, i_trace) not in feat_per_domain:
                    feat_per_domain[(i_domain, i_trace)] = []
                
                # Add the ImF value to the dictionary of values for this domain
                feat_per_domain[(i_domain, i_trace)].append(feat_val)
    
    
    # Compute the average of the values for each domain and add them to the feature array
    for (i_domain, i_trace), vals in feat_per_domain.items():
        feat_arr[i_domain, i_trace] = np.nanmean(vals)
    
    return feat_arr
