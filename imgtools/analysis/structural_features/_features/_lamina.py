import numpy as np
import trimesh
from alabtools.utils import Index

required_keys = {
    'cutoff': {'type': float, 'positive': True}
}

def run(feat_arr: np.ndarray, cell_data: dict, cell_alphashape: dict, index: Index, config: dict) -> tuple:
    """ Calculate the lamina distance and association for each spot in the cell.
    
    The lamina is taken from the alpha shape of the cell.
    
    If there are two or more spots corresponding to the same domain in the trace, the average distance is taken.

    Args:
        feat_arr (np.ndarray): initialized 0-valued array of shape (n_domains, n_traces) to store the distances
        cell_data (dict): data of the cell in dictionary format
        cell_alphashape (dict): alpha shape of the cell in dictionary format
        index (Index)
        config (dict): configuration dictionary for the lamina distance feature extraction

    Returns:
        (np.ndarray): updated array of shape (n_domains, n_traces) with the lamina distances
        (np.ndarray): updated array of shape (n_domains, n_traces) with the lamina association (1 if spot is close to lamina, 0 otherwise)
    """
    
    # Initialize the array to store the lamina associations
    feat_ass_arr = np.copy(feat_arr)
    
    # Create a counter array of same shape as feat_arr to store the number of spots per domain (for averaging)
    count_arr = np.zeros(feat_arr.shape, dtype=int)
    
    # Create a hash table for the index
    index_hash = index.get_index_hashmap()
    
    for chrom in cell_data:
            
        # Get the traces in the chromosome and hash them
        traceIDs = list(cell_data[chrom].keys())
        traceID_hash = {traceID: i for i, traceID in enumerate(traceIDs)}
        
        for traceID in cell_data[chrom]:
            
            # Get the position of the trace in the array
            i_trace = traceID_hash[traceID]
            
            for spotID in cell_data[chrom][traceID]:
                
                # Unpack the spot data
                spot_data = cell_data[chrom][traceID][spotID]
                x, y, z = spot_data['x'], spot_data['y'], spot_data['z']
                start, end = spot_data['start'], spot_data['end']
                
                # Calculate the distance to the nuclear lamina
                point = np.array([[x, y, z]])
                dist = np.abs(trimesh.proximity.signed_distance(cell_alphashape['mesh'], point)[0])
                
                # Get the position of the spot in the array using the hash tables
                i_domain = index_hash[(chrom, start, end)]
                assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
                i_domain = i_domain[0]
                
                # Increment the cell array
                feat_arr[i_domain, i_trace] += dist
                count_arr[i_domain, i_trace] += 1
                
                # Increment the cell association array
                if dist <= config['cutoff']:
                    feat_ass_arr[i_domain, i_trace] += 1
                    
    
    # Average the distances
    feat_arr = feat_arr / count_arr
    # Set to NaN the values where there are no spots
    feat_arr[count_arr == 0] = np.nan
    feat_ass_arr[count_arr == 0] = np.nan
    
    return feat_arr, feat_ass_arr
