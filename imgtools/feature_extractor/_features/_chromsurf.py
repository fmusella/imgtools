import numpy as np
import trimesh
from ...cte import ChromatinTracingExperiment
from ...cte import cte_utils
from ... import utils

docstring = """Measures the 3D distance between each spot and the surface of the chromosome territory it belongs to. 
The chromosome territory is approximated by an alpha shape fitted to the 3D points of the chromosomal trace,
and the distance is calculated as the shortest distance between the spot and the border of the shape."""

required_keys = {
    'alpha': {'type': float, 'positive': True},
    'force': {'type': str},
    'reducing_factor': {'type': float, 'positive': True},
}

def run(cellID: str, cte: ChromatinTracingExperiment, config: dict, feat_arr: np.ndarray, _) -> np.ndarray:
    """ Run the chromdepth feature extraction.
    
    For each chromosomal trace, it fits an alpha shape to the 3D points,
    and then calculates the 3D distance between each spot and the border of the shape.
    
    If there are two or more spots corresponding to the same domain in the trace, the average value is taken.

    Args:
        cellID (str)
        cte (ChromatinTracingExperiment)
        config (dict)
        feat_arr (np.ndarray): initialized 0-valued array of shape (n_domains, n_traces) to store the feature values
        _: not used, just to match the function signature
    
    Returns:
        (np.ndarray): updated array of shape (n_domains, n_traces) with the feature values
    """
    
    # Get the paramters from the config
    alpha = config['alpha']
    force = config['force']
    reducing_factor = config['reducing_factor']
    
    # Convert the force parameter to a boolean
    if force not in ['True', 'False']:
        raise ValueError(f"Error: force parameter must be 'True' or 'False', got {force}")
    force = force == 'True'
    
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
            
            # Get the position of the trace in the array
            i_trace = traceID_hash[chrom][traceID]
            
            # Get the data of the chromosomal trace in numpy format
            xs, ys, zs, _, _, _, _ = cte_utils.trace_dict_to_numpy(cell_data[chrom][traceID])
            points = np.array([xs, ys, zs]).T
            
            # If there are less than 20 points, skip this trace
            # The feature values of the spots of this trace are kept as NaN
            if len(points) < 20:
                continue
            
            # Fit the alpha shape to the points
            alpha, mesh = utils.fit_alphashape(points, alpha, force, reducing_factor)
            
            # Loop through the spots in the trace and calculate the distance to the border
            for spotID in cell_data[chrom][traceID]:
                
                # Unpack the spot data
                spot_data = cell_data[chrom][traceID][spotID]
                x, y, z = spot_data['x'], spot_data['y'], spot_data['z']
                start, end = spot_data['start'], spot_data['end']
                
                # Calculate the distance to the border
                point = np.array([[x, y, z]])
                dist = np.abs(trimesh.proximity.signed_distance(mesh, point)[0])
                
                # Get the position of the spot in the Index array using the hash tables
                i_domain = index_hash[(chrom, start, end)]
                assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
                i_domain = i_domain[0]
                
                # Add the feature value to the dictionary of values for this domain (initialize if necessary)
                if (i_domain, i_trace) not in feat_per_domain:
                    feat_per_domain[(i_domain, i_trace)] = []
                feat_per_domain[(i_domain, i_trace)].append(dist)
    
    # Compute the average of the values for each domain and add them to the feature array
    for (i_domain, i_trace), vals in feat_per_domain.items():
        feat_arr[i_domain, i_trace] = np.nanmean(vals)
    
    return feat_arr
