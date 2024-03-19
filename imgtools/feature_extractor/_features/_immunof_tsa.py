import os
import numpy as np
from scipy.spatial.distance import cdist
import h5py
from alabtools.utils import Index
from ...cte import cte_utils

required_keys = {
    'ImF_file': {'type': str},
    'tsa_alpha': {'type': (float), 'positive': True},
    'top_percentile': {'type': (float), 'positive': True}
}

def run(cellID: str, feature: str, feat_arr: np.ndarray, cell_data: dict, index: Index, config: dict) -> np.ndarray:
    """ Get the simulated single-cell TSA experiment from an ImmunoFluorescence (ImF) file.
    
    The TSA-distance is calculated as the sum of the exponential of the negative distance between each spot
    and the spot with the highest ImF value:
           TSA[i] = sum_j exp(-alpha * dist[i,j])
    
    The ImF values are stored in the HDF5 file that is specified in the configuration file.
    
    If multiple spots are associated with the same domain, the median of the ImF-TSA values is taken.

    Args:
        cellID (str)
        feature (str): Name of the feature to extract
        feat_arr (np.ndarray): initialized 0-valued array of shape (ndomain, max_ntrace_per_chrom) to store the feature values
        cell_data (dict): Data of the cell in dictionary format
        index (Index)
        config (dict): Configuration dictionary for the feature extraction

    Returns:
        np.ndarray: Updated array of shape (n_domains, n_traces) with the feature values
    """
    
    # If the feature name ends with '_tsa', remove it
    if feature.endswith('_tsa'):
        feature = feature[:-4]
    
    # Check that the ImF file exists
    if not os.path.isfile(config['ImF_file']):
        raise ValueError(f"The ImmunoFluorescence file {config['ImF_file']} does not exist.")
    
    # Get the ImmunoFluorescence HDF5 file, raise an error if h5py can't open it
    try:
        imf_h5 = h5py.File(config['ImF_file'], 'r')
    except Exception as e:
        raise ValueError(f"Error opening the ImmunoFluorescence file: {e}")
    
    # Make sure the feature is in the HDF5 file
    if feature not in imf_h5[cellID]:
        raise ValueError(f"Feature {feature} not found in the HDF5 file for cell {cellID}")
    
    # Get the TSA exponent from the configuration file
    tsa_alpha = config['tsa_alpha']
    
    # Get the data - a numpy array of shape (nspot,) - for this feature in the cell
    imf_vals = imf_h5[cellID][feature][:]
    # Get the spotIDs associated with the imf array
    imf_spotIDs = imf_h5[cellID]['spotIDs'][:].astype('U20')
    imf_h5.close()
    
    # Get the spotIDs with the highest ImF values
    imf_top_idx = np.where(imf_vals > np.percentile(imf_vals, config['top_percentile']))[0]
    imf_top_spotIDs = imf_spotIDs[imf_top_idx]
    
    # Convert the cell_data from dict to numpy format
    xs, ys, zs, _, _, _, _, _, spotIDs = cte_utils.cell_dict_to_numpy(cell_data)
    
    # Get the coordinates of the top ImF spots
    idx_top = np.where(np.isin(spotIDs, imf_top_spotIDs))[0]
    crd_top = np.column_stack((xs[idx_top], ys[idx_top], zs[idx_top]))
    
    del imf_vals, imf_spotIDs, imf_top_idx, imf_top_spotIDs, xs, ys, zs, spotIDs, idx_top
    
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
            
            # Get the position of the trace in the array using the hash tables
            i_trace = traceID_hash[traceID]
            
            for spotID in cell_data[chrom][traceID]:
                
                # Unpack the spot data
                spot_data = cell_data[chrom][traceID][spotID]
                x, y, z = spot_data['x'], spot_data['y'], spot_data['z']
                start, end = spot_data['start'], spot_data['end']
                
                # Get the position of the spot in the array using the hash tables
                i_domain = index_hash[(chrom, start, end)]
                # Make sure that there is only one idx for this domain in the Index (i.e. it's haploid)
                assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
                i_domain = i_domain[0]
                
                # Calculate the TSA distance for this spot
                dists = cdist(np.array([[x, y, z]]), crd_top).flatten()
                tsa_dist = np.sum(np.exp(- tsa_alpha * dists))
                
                # Add the TSA distance to the dictionary of values for this domain (initialize if necessary)
                if (i_domain, i_trace) not in feat_per_domain:
                    feat_per_domain[(i_domain, i_trace)] = []
                feat_per_domain[(i_domain, i_trace)].append(tsa_dist)
    
    # Compute the median of the values for each domain and add them to the feature array
    for (i_domain, i_trace), vals in feat_per_domain.items():
        feat_arr[i_domain, i_trace] = np.median(vals)
    
    return feat_arr
