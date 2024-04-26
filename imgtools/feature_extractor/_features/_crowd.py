import numpy as np
from scipy.spatial.distance import cdist
from ...cte import ChromatinTracingExperiment
from ...cte import cte_utils

docstring = """Measure the local crowdiness of each spot in the cell.
There are two methods to measure the crowdiness:
- Density: the density of spots within a sphere of a given radius centered at the spot.
- Average: the average distance between the spot and all other spots within a sphere of a given radius centered at the spot of interest.
For the average method, if there are no other spots within the sphere, the feature value is set to NaN."""

required_keys = {
    'method': {'type': str},
    'radius': {'type': float, 'positive': True},
}

AVAILABLE_METHODS = ['density', 'average']

def run(cellID: str, cte: ChromatinTracingExperiment, config: dict, feat_arr: np.ndarray, _) -> np.ndarray:
    """ For each spot, measure the local crowdiness.
    
    There are two methods to measure the crowdiness:
    - Density: the number of spots within a sphere centered at the spot.
    - Average: the average distance of the spots with all other spots within a sphere centered at the spot of interest.
    
    If two or more spots are mapped to the same domain, the average of the values is taken.

    Args:
        cellID (str)
        cte (ChromatinTracingExperiment)
        config (dict)
        feat_arr (np.ndarray): initialized 0-valued array of shape (n_domains, n_traces) to store the feature values
        _: not used, just to match the function signature

    Returns:
        (np.ndarray): updated array of shape (n_domains, n_traces) with the feature values
    """
    
    # Get the parameters from the configuration
    method = config['method']
    radius = config['radius']
    
    # Check that the method is valid
    if method not in AVAILABLE_METHODS:
        raise ValueError(f"Error: method '{method}' not recognized. Available methods: {', '.join(AVAILABLE_METHODS)}")
    
    # Get the cell data in dictionary format
    cell_data = cte.get_data(cellID)
    
    # Get the traceID hash table to map traces to their position in the array
    traceID_hash = cte.get_trace_hashmap(cellID)
    
    # Convert the cell data in numpy format and get the coordinates of each spot
    xs, ys, zs, _, _, _, _, _, _ = cte_utils.cell_dict_to_numpy(cell_data)
    crds = np.array([xs, ys, zs]).T
    
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

                # Calculate the distance of this spot to all the other spots
                point = np.array([[x, y, z]])
                dists = cdist(point, crds).flatten()
                
                # Get the array of distances smaller than the radius
                dists_in_sphere = dists[dists < radius]
                
                # Remove the distance to the spot itself
                dists_in_sphere = dists_in_sphere[dists_in_sphere > 0]
                
                # Calculate the crowdiness value
                if method == 'density':
                    # Calculate the density of points within the sphere
                    feat_val = len(dists_in_sphere) / (4/3 * np.pi * radius**3)
                elif method == 'average':
                    # If there are no other spots within the sphere, skip this spot (value kept as NaN)
                    if len(dists_in_sphere) == 0:
                        continue
                    # Otherwise, calculate the average distance to the other spots
                    else:
                        feat_val = np.mean(dists_in_sphere)
                
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
