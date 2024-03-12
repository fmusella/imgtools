import numpy as np
import trimesh
from alabtools.utils import Index

required_keys = {}

def run(feat_arr: np.ndarray, cell_data: dict, cell_alphashape: dict, index: Index) -> tuple:
    """ Calculate the distance of each spot to the nuclear envelope.
    
    The nuclear envelope is taken from the alpha shape of the cell.
    
    If there are two or more spots corresponding to the same domain in the trace, the average distance is taken.

    Args:
        feat_arr (np.ndarray): initialized 0-valued array of shape (n_domains, n_traces) to store the distances
        cell_data (dict): data of the cell in dictionary format
        cell_alphashape (dict): alpha shape of the cell in dictionary format
        index (Index)

    Returns:
        (np.ndarray): updated array of shape (n_domains, n_traces) with the distances to the nuclear envelope
    """
    
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
            
            # Get the position of the trace in the array
            i_trace = traceID_hash[traceID]
            
            for spotID in cell_data[chrom][traceID]:
                
                # Unpack the spot data
                spot_data = cell_data[chrom][traceID][spotID]
                x, y, z = spot_data['x'], spot_data['y'], spot_data['z']
                start, end = spot_data['start'], spot_data['end']
                
                # Calculate the distance to the nuclear envelope
                point = np.array([[x, y, z]])
                dist = np.abs(trimesh.proximity.signed_distance(cell_alphashape['mesh'], point)[0])
                
                # Get the position of the spot in the Index array using the hash tables
                i_domain = index_hash[(chrom, start, end)]
                assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
                i_domain = i_domain[0]
                
                # Add the feature value to the dictionary of values for this domain (initialize if necessary)
                if (i_domain, i_trace) not in feat_per_domain:
                    feat_per_domain[(i_domain, i_trace)] = []
                feat_per_domain[(i_domain, i_trace)].append(dist)
                
    
    # Compute the median of the values for each domain and add them to the feature array
    for (i_domain, i_trace), vals in feat_per_domain.items():
        feat_arr[i_domain, i_trace] = np.median(vals)
    
    return feat_arr
