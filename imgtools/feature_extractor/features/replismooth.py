import os
import h5py
import numpy as np
from scipy.spatial.distance import cdist
from ...cte import ChromatinTracingExperiment
from ...cte import cte_utils

docstring = """Perform a 3D smoothing of the RepliProb matrix using a sphere of a given radius around each spot.
The RepliProb matrix is taken from the SimulatedRepliSeq h5py file."""

required_keys = {
    'simrep_file': {'type': str},
    'radius': {'type': float, 'positive': True}
}

def run(cellID: str, cte: ChromatinTracingExperiment, config: dict, feat_arr: np.ndarray, _) -> np.ndarray:
    """ Perform a 3D smoothing of the RepliProb matrix using a sphere of a given radius around each spot.
    
    The RepliProb matrix is taken from the SimulatedRepliSeq h5py file.

    Args:
        cellID (str)
        cte (ChromatinTracingExperiment)
        config (dict): configuration dictionary with the following keys:
            - 'simrep_file' (str): path to the SimulatedRepliSeq file
            - 'radius' (float): radius of the sphere to use for the smoothing
        feat_arr (np.ndarray): initialized nan-valued array of shape (n_domains, n_traces) to store the feature values
        _: not used, just to match the function signature

    Returns:
        (np.ndarray): updated array of shape (n_domains, n_traces) with the feature values
    """
    
    # Get the radius of the sphere to use for the smoothing
    radius = config['radius']
    
    # Check that the SimulatedRepliSeq file exists
    file = config['simrep_file']
    if not os.path.isfile(file):
        raise ValueError(f"The SimulatedRepliSeq file {file} does not exist")
    
    # Get the SimulatedRepliSeq HDF5 file, raise an error if h5py can't open it
    try:
        h5 = h5py.File(file, 'r')
    except Exception as e:
        raise ValueError(f"Error opening the SimulatedRepliSeq file as HDF5 file: {e}")
    
    # Get the RepliProb matrix 
    try:
        cellnum = cte.get_cellnum(cellID)
        repli_arr = h5['p_ic_SW'][cellnum, :, :]
    except Exception as e:
        raise ValueError(f"Error reading the RepliProb matrix from the SimulatedRepliSeq file: {e}")
    
    # Make sure the RepliProb matrix has the same shape as the feature array
    if repli_arr.shape != feat_arr.shape:
        raise ValueError(f"The RepliProb matrix shape {repli_arr.shape} does not match the feature array shape {feat_arr.shape}")
    
    # Get the cell data in dictionary format
    cell_data = cte.get_data(cellID)
    
    # Get the traceID hash table to map traces to their position in the array
    traceID_hash = cte.get_trace_hashmap(cellID)
    
    # Convert the cell data in numpy format and get the coordinates of each spot
    d = cte_utils.cell_dict_to_numpy(cell_data)
    xs, ys, zs, chroms, starts, ends, traceIDs = d['xs'], d['ys'], d['zs'], d['chroms'], d['starts'], d['ends'], d['traceIDs']
    crds = np.array([xs, ys, zs]).T
    
    # Get the index and its hash table
    index = cte.index
    index_hash = index.get_index_hashmap()
    
    # Map the RepliProb matrix to the CTE
    replis = map_featmat_to_cte(cellID, repli_arr, index_hash, traceID_hash, traceIDs, chroms, starts, ends)
    
    # Initialize a dictionary to store the feature values for each domain (we will then take the average)
    feat_per_domain = {}
    
    for chrom in cell_data:       
        for traceID in cell_data[chrom]:
            
            # Get the position of the trace in the array
            i_trace = traceID_hash[chrom][traceID]
            
            for spotID in cell_data[chrom][traceID]:
                
                # Unpack the spot data
                spot_data = cell_data[chrom][traceID][spotID]
                x, y, z = spot_data['x'], spot_data['y'], spot_data['z']
                start, end = spot_data['start'], spot_data['end']

                # Calculate the distance of this spot to all the other spots
                point = np.array([[x, y, z]])
                dists = cdist(point, crds)[0]
                
                # Get the array of RepliProbs smaller than the radius
                replis_in_sphere = replis[dists < radius]
                
                # Calculate the average RepliProb in the sphere
                feat_val = np.nanmean(replis_in_sphere)
                
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


def map_featmat_to_cte(
    cellID: str,
    feat_arr: np.ndarray,
    index_hash: dict,
    traceID_hash: dict,
    traceIDs: np.ndarray,
    chroms: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """ Get the feature values for a cell in the same order as the spots in the CTE.

    Args:
        cellID (str)
        feat_arr (np.ndarray): Feature matrix for the cell, shape (n_domains, n_traces)
        index_hash (dict): Dictionary that maps domains (chrom, start, end) to their position in the index array
        traceID_hash (dict): Dictionary that maps traceIDs to numpy array indices, obtained from the CTE
        traceIDs (np.ndarray): Array of traceIDs for the spots
        chroms (np.ndarray): Array of chromosome names for the spots
        starts (np.ndarray): Array of start positions for the spots
        ends (np.ndarray): Array of end positions for the spots

    Returns:
        featvals (np.ndarray): Array of feature values for the spots, ordered as the spots in the CTE
    """
    
    # Get the feature values for the cell, in the same order as the spots
    featvals = []
    for traceID, chrom, start, end in zip(traceIDs, chroms, starts, ends):
        
        # Get the position of the spot in the array using the hash tables
        i_domain = index_hash[(chrom, start, end)]
        assert len(i_domain) == 1, f"Multiple domains found for {chrom}:{start}-{end} in cell {cellID}."
        i_domain = i_domain[0]
        i_trace = traceID_hash[chrom][traceID]
        
        # Get the feature value
        featval = feat_arr[i_domain, i_trace]
        featvals.append(featval)
    featvals = np.array(featvals).astype(float)
    
    return featvals
