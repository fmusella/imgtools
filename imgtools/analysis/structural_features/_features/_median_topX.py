import numpy as np
from scipy.spatial.distance import cdist
from alabtools.utils import Index
from ....cte import cte_utils

required_keys = {
    'npoint_to_median': {'type': int, 'positive': True},
}

def run(feat_arr: np.ndarray, cell_data: dict, index: Index, config: dict) -> tuple:
    """ For each spot, calculate the median distance to the top X (e.g. top 10) closest spots in the cell.
    
    To make the calculation more efficient, we first find a neighborhood box around each spot that should contain at least 10 spots.
    This box is obtained by finding the maximum x-x, y-y, z-z distances across all spots and assuming that a box with 20% of these distances will do the job.
    We then filter out spots that are too far from the spot and for sure are not in the top X.
    We then calculate the distance of each point in the box to the spot and take the top X closest spots.
    
    If there are not enough points in the box, the feature value is set to the diagonal of the box.
    
    If two or more spots are mapped to the same domain, the median value is taken.

    Args:
        feat_arr (np.ndarray): initialized 0-valued array of shape (n_domains, n_traces) to store the distances
        cell_data (dict): data of the cell in dictionary format
        cell_alphashape (dict): alpha shape of the cell in dictionary format
        index (Index)
        config (dict): configuration dictionary

    Returns:
        (np.ndarray): updated array of shape (n_domains, n_traces) with the distances to the top X closest spots
    """
    
    # Get the cell data in dictionary format and get the coordinates of each spot
    xs, ys, zs, _, _, _, _, _, _ = cte_utils.cell_dict_to_numpy(cell_data)
    crds = np.array([xs, ys, zs]).T
    
    # Find neighborhood box size
    # This is a virtual box around each spot that should contain at least 10 spots
    # We are going to use this box to filter out spots that are too far from the spot and for sure are not in the top X
    # To find this box, we first find the maximum x-x, y-y, z-z distances across all spots
    x_max = np.max(xs) - np.min(xs)
    y_max = np.max(ys) - np.min(ys)
    z_max = np.max(zs) - np.min(zs)
    # We are going to assume that a box with 20% of these distances will do the job
    x_box = x_max * 0.2
    y_box = y_max * 0.2
    z_box = z_max * 0.2
    
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
                
                # Get the position of the spot in the Index array using the hash tables
                i_domain = index_hash[(chrom, start, end)]
                assert len(i_domain) == 1, f"Error: multiple domains found for {chrom}, {start}, {end}"
                i_domain = i_domain[0]
                
                # Initialize the list of values for this domain if necessary
                if (i_domain, i_trace) not in feat_per_domain:
                    feat_per_domain[(i_domain, i_trace)] = []
                
                # Isolate the points that are within the neighborhood box centered at the spot
                idx_inbox_x = np.where(np.abs(xs - x) < x_box/2)[0]
                idx_inbox_y = np.where(np.abs(ys - y) < y_box/2)[0]
                idx_inbox_z = np.where(np.abs(zs - z) < z_box/2)[0]
                idx_inbox = np.intersect1d(np.intersect1d(idx_inbox_x, idx_inbox_y), idx_inbox_z)
                
                # If there are too few points in the box, set the feature value to the diagonal of the box
                if len(idx_inbox) < config['npoint_to_median'] + 1:
                    raise ValueError(f"Error: not enough points in the box for spot {spotID} in domain {i_domain}")
                    feat_per_domain[(i_domain, i_trace)].append(np.sqrt(x_box**2 + y_box**2 + z_box**2))
                    continue
                
                # Get the coordinates of the points in the box
                crds_inbox = crds[idx_inbox, :]
                
                # Calculate the distance of each point in the box to the spot
                point = np.array([[x, y, z]])
                dists_inbox = cdist(point, crds_inbox).flatten()
                
                # Take the top X smallest distances
                # (add 1 to the number of points to take the spot itself into account, to be removed later)
                npoint = config['npoint_to_median'] + 1
                dists_top = np.partition(dists_inbox, npoint - 1)[:npoint]
                dists_top = dists_top[dists_top > 0]  # remove self-distance
                
                # Take the median of the distances
                median_dist = np.median(dists_top)
                
                # Add the feature value to the dictionary of values for this domain (initialize if necessary)
                feat_per_domain[(i_domain, i_trace)].append(median_dist)
                
    
    # Compute the median of the values for each domain and add them to the feature array
    for (i_domain, i_trace), vals in feat_per_domain.items():
        feat_arr[i_domain, i_trace] = np.median(vals)
    
    return feat_arr
