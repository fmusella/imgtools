import numpy as np
from scipy.spatial.distance import cdist
from ...cte import ChromatinTracingExperiment
from ...cte import cte_utils

docstring = """Measures the inter or long-range-intra contact probability within a sphere centered at each spot."""

required_keys = {
    'range': {'type': str},
    'radius': {'type': float, 'positive': True},
}

AVAILABLE_RANGES = ['inter', 'long_intra']

def run(cellID: str, cte: ChromatinTracingExperiment, config: dict, feat_arr: np.ndarray, _) -> np.ndarray:
    """ For each spot, measures the probability of having a far-range contact (either inter or long-range intra).
    The contact is defined as proximity within a sphere of a given radius centered at the spot.
    
    If two or more spots are mapped to the same domain, the average of the values is taken.

    Args:
        cellID (str)
        cte (ChromatinTracingExperiment)
        config (dict): configuration dictionary
        feat_arr (np.ndarray): initialized nan-valued array of shape (n_domains, n_traces) to store the feature value.
        _: not used, just to match the function signature

    Returns:
        (np.ndarray): updated array of shape (n_domains, n_traces) with the feature values.
    """
    
    # Get the parameters from the configuration
    range = config['range']
    radius = config['radius']
    
    # Check that the range is valid
    if range not in AVAILABLE_RANGES:
        raise ValueError(f"Error: range '{range}' not recognized. Available ranges: {', '.join(AVAILABLE_RANGES)}")
    
    # Get the genomic-distance threshold for long-range intra-chromosomal contacts
    # If it's in the coniguration, use it.
    # Otherwise, use the resolution of the index and define it as 10 times that.
    if range == 'long_intra':
        try:
            long_intra_threshold = config['long_intra_threshold']
        except KeyError:
            resolution = cte.index.resolution()
            if resolution is None:
                raise ValueError("Error: resolution of the index not found. Please provide 'long_intra_threshold' in the configuration.")
            long_intra_threshold = int(10 * resolution)
    
    # Get the cell data in dictionary format
    cell_data = cte.get_data(cellID)
    
    # Get the traceID hash table to map traces to their position in the array
    traceID_hash = cte.get_trace_hashmap(cellID)
    
    # Convert the cell data in numpy format and get the coordinates of each spot
    xs, ys, zs, chroms, starts, _, _, _, _ = cte_utils.cell_dict_to_numpy(cell_data)
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
                
                # Get the mask to filter the spots within the contact sphere
                mask_in_sphere = dists < radius
                
                # Get the total number of contacts of the spot (excluding itself)
                ncontacts = np.sum(mask_in_sphere) - 1
                
                # If there are no contacts, skip this spot (the feature value is kept as NaN)
                if ncontacts == 0:
                    continue
                
                # Get the chromosome ID of the spots within the contact sphere
                chroms_in_sphere = chroms[mask_in_sphere]
                # Get the start position of the spots within the contact sphere
                starts_in_sphere = starts[mask_in_sphere]
         
                # Calculate the long-range contact probability (inter or long-intra)
                if range == 'inter':
                    # Calculate the ratio of inter-chromosomal contacts to the total number of contacts
                    ncontacts_inter = np.sum(chroms_in_sphere != chrom)
                    feat_val = ncontacts_inter / ncontacts
                elif range == 'long_intra':
                    # Calculate the number of long-range intra-chromosomal contacts
                    start_in_sphere_intra = starts_in_sphere[chroms_in_sphere == chrom]
                    ncontacts_long_intra = np.sum(np.abs(start_in_sphere_intra - start) >= long_intra_threshold)
                    feat_val = ncontacts_long_intra / ncontacts
                
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
