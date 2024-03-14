import numpy as np
from scipy.spatial.distance import cdist
from alabtools.utils import Index
from ....cte import cte_utils

required_keys = {
    'radius': {'type': float, 'positive': True},
}

def run(feat_arr: np.ndarray, cell_data: dict, index: Index, config: dict) -> tuple:
    """ For each spot, calculate the density of points within a sphere of a given radius.
    
    If two or more spots are mapped to the same domain, the median density is taken.

    Args:
        feat_arr (np.ndarray): initialized 0-valued array of shape (n_domains, n_traces) to store the densities.
        cell_data (dict): data of the cell in dictionary format
        index (Index)
        config (dict): configuration dictionary

    Returns:
        (np.ndarray): updated array of shape (n_domains, n_traces) with the densities.
    """
    
    # Get the radius of the sphere
    try:
        radius = config['radius']
    except KeyError:
        raise KeyError("Error: 'radius' not found in the configuration dictionary")
    
    # Get the cell data in dictionary format and get the coordinates of each spot
    xs, ys, zs, _, _, _, _, _, _ = cte_utils.cell_dict_to_numpy(cell_data)
    crds = np.array([xs, ys, zs]).T
    
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

                # Calculate the distance of each point to the spot
                point = np.array([[x, y, z]])
                dists = cdist(point, crds).flatten()
                
                # Get the number of points within the sphere of the given radius
                npoints = np.sum(dists < radius)
                
                # Calculate the density
                density = npoints / (4/3 * np.pi * radius**3)
                
                # Get the position of the spot in the Index array using the hash tables
                i_domain = index_hash[(chrom, start, end)]
                assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
                i_domain = i_domain[0]
                
                # Initialize the list of values for this domain if necessary
                if (i_domain, i_trace) not in feat_per_domain:
                    feat_per_domain[(i_domain, i_trace)] = []
                
                # Add the feature value to the dictionary of values for this domain (initialize if necessary)
                feat_per_domain[(i_domain, i_trace)].append(density)
                
    
    # Compute the median of the values for each domain and add them to the feature array
    for (i_domain, i_trace), vals in feat_per_domain.items():
        feat_arr[i_domain, i_trace] = np.median(vals)
    
    return feat_arr
