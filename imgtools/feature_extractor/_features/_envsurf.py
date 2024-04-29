import numpy as np
import trimesh
from ...cte import ChromatinTracingExperiment

docstring = """Measures the 3D distance between each spot and the nuclear envelope. 
The nuclear envelope is taken from the alpha shape of the cell, 
and the distance is calculated as the shortest distance between the spot and the alpha shape surface."""

required_keys = {}

def run(cellID: str, cte: ChromatinTracingExperiment, _1, feat_arr: np.ndarray, _2) -> np.ndarray:
    """ Calculate the distance of each spot to the nuclear envelope.
    
    The nuclear envelope is taken from the alpha shape of the cell.
    
    If there are two or more spots corresponding to the same domain in the trace, the average distance is taken.

    Args:
        cellID (str)
        cte (ChromatinTracingExperiment)
        feat_arr (np.ndarray): initialized 0-valued array of shape (n_domains, n_traces) to store the feature values
        _*: not used, just to match the function signature

    Returns:
        (np.ndarray): updated array of shape (n_domains, n_traces) with the feature values
    """
    
    # Get the cell data in dictionary format
    cell_data = cte.get_data(cellID)
    
    # Get the alpha shape of the cell
    cell_alphashape = cte.get_alphashapes(cellID)
    
    # Get the traceID hash table to map traces to their position in the array
    traceID_hash = cte.get_trace_hashmap(cellID)
    
    # Get the index and its hash table
    index = cte.index
    index_hash = index.get_index_hashmap()
    
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
                
                # Calculate the distance to the nuclear envelope
                point = np.array([[x, y, z]])
                dist = np.abs(trimesh.proximity.signed_distance(cell_alphashape['mesh'], point)[0])
                
                # Get the position of the spot in the Index array using the hash tables
                i_domain = index_hash[(chrom, start, end)]
                assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
                i_domain = i_domain[0]
                
                # Initialize the list of values for this domain if necessary
                if (i_domain, i_trace) not in feat_per_domain:
                    feat_per_domain[(i_domain, i_trace)] = []
                
                # Add the feature value to the dictionary of values for this domain
                feat_per_domain[(i_domain, i_trace)].append(dist)
                
    
    # Compute the average of the values for each domain and add them to the feature array
    for (i_domain, i_trace), vals in feat_per_domain.items():
        feat_arr[i_domain, i_trace] = np.nanmean(vals)
    
    return feat_arr
