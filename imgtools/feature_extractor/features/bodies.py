import os
import numpy as np
import h5py
from scipy.spatial.distance import cdist
from ...cte import ChromatinTracingExperiment

docstring = """..."""

required_keys = {
    'bodies_file': {'type': str},
    'body': {'type': str}
}

def run(cellID: str, cte: ChromatinTracingExperiment, config: dict, feat_arr: np.ndarray, _) -> np.ndarray:
    """ Summary...

    Args:
        cellID (str)
        cte (ChromatinTracingExperiment)
        config (dict): configuration dictionary with the following keys:
            - 'bodies_file' (str): path to the HDF5 file containing the bodies data
            - 'body' (str): name of the body to use
        feat_arr (np.ndarray): initialized nan-valued array of shape (n_domains, n_traces) to store the feature values
        _*: not used, just to match the function signature

    Returns:
        (np.ndarray): updated array of shape (n_domains, n_traces) with the feature values
    """
    
    # Unpack the config dictionary
    bodies_h5_file = config['bodies_file']
    body = config['body']
    
    # Check that the Bodies file exists
    if not os.path.isfile(bodies_h5_file):
        raise ValueError(f"The Bodies file {bodies_h5_file} does not exist.")
    
    # Get the Bodies HDF5 file, raise an error if h5py can't open it
    try:
        bodies_h5 = h5py.File(bodies_h5_file, 'r')
    except Exception as e:
        raise ValueError(f"Error opening the Bodies file as HDF5 file: {e}")
    
    # Make sure the cell is in the HDF5 file
    if cellID not in bodies_h5:
        raise ValueError(f"Cell {cellID} not found in the Bodies HDF5 file")
    
    # Make sure the body is in the HDF5 file
    if body not in bodies_h5[cellID]:
        raise ValueError(f"Body {body} not found in the Bodies HDF5 file for cell {cellID}")
    
    # Get the body data for the cell:
    #    1) res (float) - voxel resolution of the image
    #    2) origin (np.ndarray[3]) - XYZ origins of the body's bounding box, in unit of voxels
    #    3) image (np.ndarray) - binary image of the body
    res = bodies_h5.attrs['resolution']
    origin = bodies_h5[cellID]['origin'][:]  # shape (3,)
    image = bodies_h5[cellID][body]['bimage'][:].astype(int)  # shape (n_x, n_y, n_z)
    
    # Close the HDF5 file
    bodies_h5.close()
    
    # Get the indices of the body voxels, i.e. the foreground
    body_indices = np.argwhere(image)
    
    # If there are no body voxels, return the array as it is
    if body_indices.size == 0:
        return feat_arr
    
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
                x, y, z = spot_data['x'], spot_data['y'], spot_data['z']
                start, end = spot_data['start'], spot_data['end']
                
                # Map the 3D coordinates to the voxel coordinates
                # (we rescale by the voxel resolution and subtract the origin)
                crd = np.array([x, y, z]) / res - origin
                # Interpolate to closest integer coordinates
                crd = np.round(crd).astype(int)
                
                # Calculate the distances to all the body voxels
                dists = cdist([crd], body_indices, metric='euclidean').flatten()
                
                # Convert the distances to physical units
                dists = dists * res
                
                # Get the minimum distance
                dist = np.min(dists)
                
                # Get the position of the spot in the array using the hash tables
                i_domain = index_hash[(chrom, start, end)]
                assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
                i_domain = i_domain[0]
                
                # Initialize the list of values for this domain if necessary
                if (i_domain, i_trace) not in feat_per_domain:
                    feat_per_domain[(i_domain, i_trace)] = []
                
                # Add the distance value to the dictionary of values for this domain
                feat_per_domain[(i_domain, i_trace)].append(dist)
    
    
    # Compute the average of the values for each domain and add them to the feature array
    for (i_domain, i_trace), vals in feat_per_domain.items():
        feat_arr[i_domain, i_trace] = np.nanmean(vals)
    
    return feat_arr
