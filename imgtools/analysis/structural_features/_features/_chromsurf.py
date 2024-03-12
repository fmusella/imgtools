import numpy as np
import trimesh
from alabtools.utils import Index
from ....cte import cte_utils
from .... import utils

required_keys = {
    'alpha': {'type': float, 'positive': True},
    'force': {'type': bool},
    'reducing_factor': {'type': float, 'positive': True},
}

def run(feat_arr: np.ndarray, cell_data: dict, index: Index, config: dict):
    """ Run the chromdepth feature extraction.
    
    For each chromosomal trace, it fits an alpha shape to the 3D points,
    and then calculates the 3D distance between each spot and the border of the shape.
    
    If there are two or more spots corresponding to the same domain in the trace, the average distance is taken.

    Args:
        feat_arr (np.ndarray): initialized 0-valued array of shape (n_domains, n_traces) to store the distances
        cell_data (dict): cell data in dictionary format
        index (Index)
        config (dict): configuration for the feature
    
    Returns:
        (np.ndarray): updated array of shape (n_domains, n_traces) with the chromosome surface distances
    """
    
    # Create a counter array of same shape as feat_arr to store the number of spots per domain (for averaging)
    count_arr = np.zeros(feat_arr.shape, dtype=int)
    
    # Create a hash table for the index
    index_hash = index.get_index_hashmap()
    
    for chrom in cell_data:
            
        # Get the traces in the chromosome and hash them
        traceIDs = list(cell_data[chrom].keys())
        traceIDs.sort()  # Sort to ensure that the order doesn't depend on how the dictionary is iterated
        traceID_hash = {traceID: i for i, traceID in enumerate(traceIDs)}
        
        for traceID in cell_data[chrom]:
            
            # Get the position of the trace in the array
            i_trace = traceID_hash[traceID]
            
            # Get the data of the chromosomal trace in numpy format
            xs, ys, zs, _, _, _, _ = cte_utils.trace_dict_to_numpy(cell_data[chrom][traceID])
            points = np.array([xs, ys, zs]).T
            
            # Fit the alpha shape to the points
            alpha, mesh = utils.fit_alphashape(points, config['alpha'], config['force'], config['reducing_factor'])
            
            # Loop through the spots in the trace and calculate the distance to the border
            for spotID in cell_data[chrom][traceID]:
                
                # Unpack the spot data
                spot_data = cell_data[chrom][traceID][spotID]
                x, y, z = spot_data['x'], spot_data['y'], spot_data['z']
                start, end = spot_data['start'], spot_data['end']
                
                # Calculate the distance to the border
                point = np.array([[x, y, z]])
                dist = np.abs(trimesh.proximity.signed_distance(mesh, point)[0])
                
                # Get the position of the spot in the array using the hash tables
                i_domain = index_hash[(chrom, start, end)]
                assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
                i_domain = i_domain[0]
                
                # Increment the feature and the count array
                feat_arr[i_domain, i_trace] += dist
                count_arr[i_domain, i_trace] += 1
    
    # Average the distances
    feat_arr = feat_arr / count_arr
    # Set to NaN the values where there are no spots
    feat_arr[count_arr == 0] = np.nan
    
    return feat_arr
