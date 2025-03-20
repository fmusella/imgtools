import os
import numpy as np
from scipy.spatial.distance import cdist
from ...cte import ChromatinTracingExperiment
from ...cte import cte_utils

docstring = """For each spot, it measures the Gaussian Kernel Density contributions coming from all the other spots.
If provided, the Kernel Density contributions are calculated only from a subset of domains defined in a BED file.
Reference for Weighted Kernel Density:
    Hall, P & Huang, LS (2002), 'Unimodal density estimation using kernel methods', Statistica Sinica, 12, 965-990.
The Kernel Density is calculated as the sum of Gaussian densities centered at each spot, with a given bandwidth (sigma):
    K_i = (1 / N) * (1 / ((2 * pi)^1.5 * sigma^3) * sum_j [w_j * exp(-||x_i - x_j||^2 / (2 * sigma^2))],
where:
    - K_i is the Kernel Density at spot i,
    - N is the number of spots used to calculate the Kernel Density,
    - the sum is over spots j different from i: either all spots or only from a subset of domains,
    - sigma is the bandwidth of the Gaussian Kernel Density,
    - x_i and x_j are the coordinates of spots i and j,
    - ||x_i - x_j|| is the Euclidean distance between spots i and j,
    - w_j is the weight of the distance to the spot j (1 if not provided)."""

required_keys = {
    'sigma': {'type': float, 'positive': True},
}

def kernel_density(dists: np.ndarray, sigma: float, weights: np.array = None) -> float:
    """ Calculate the Kernel Density for a set of distances:
        K = (1 / N) * (1 / ((2 * pi)^1.5 * sigma^3)) * sum_j [w_j * e^(-d_j^2 / (2 * sigma^2))],
    where:
        - K is the Kernel Density,
        - N is the total number of distances,
        - d_j is the j-th distance (corresponding to ||x_i - x_j||),
        - sigma is the bandwidth of the Gaussian Kernel Density,
        - w_j is the j-th weight (1 if not provided).
    
    Reference for Weighted Kernel Density:
        Hall, P & Huang, LS (2002), 'Unimodal density estimation using kernel methods', Statistica Sinica, 12, 965-990.

    Args:
        dists (np.ndarray): array of distances.
        sigma (float): bandwidth of the Gaussian Kernel Density.
        weights (np.array, optional): array of weights for each distance. Default is None.

    Returns:
        float: the Kernel Density value.
    """
    # If weights are not provided, set them to 1
    if weights is None:
        weights = np.ones(len(dists))
    # Calculate the Kernel Density
    kd = (1 / len(dists)) * (1 / ((2 * np.pi)**1.5 * sigma**3)) * np.sum(weights * np.exp(-dists**2 / (2 * sigma**2)))
    return kd

def run(cellID: str, cte: ChromatinTracingExperiment, config: dict, feat_arr: np.ndarray, _) -> np.ndarray:
    """ For each spot, it measures the Gaussian Kernel Density contributions coming from other spots.
    
    If provided, the Kernel Density contributions are calculated only from a subset of domains defined in a BED file.
    Otherwise, all the other spots are considered.
    
    The Kernel Density is calculated as the sum of Gaussian densities centered at each spot, with a given bandwidth (sigma):
        K_i = (1 / N) * (1 / ((2 * pi)^1.5 * sigma^3) * sum_j [w_j * exp(-||x_i - x_j||^2 / (2 * sigma^2))],
    where:
        - K_i is the Kernel Density at spot i,
        - N is the number of spots used to calculate the Kernel Density,
        - the sum is over spots j different from i: either all spots or only from a subset of domains,
        - x_i and x_j are the coordinates of spots i and j,
        - ||x_i - x_j|| is the Euclidean distance between spots i and j,
        - w_j is the weight of the distance to the spot j (1 if not provided).
    
    If two or more spots are mapped to the same domain, the average of the values is taken.

    Args:
        cellID (str)
        cte (ChromatinTracingExperiment)
        config (dict): configuration dictionary with the following keys:
            - sigma (float): bandwidth of the Gaussian Kernel Density.
            - bedfile (str, optional): path to the BED file containing
                    the list of domains to consider for the Kernel Density.
            - weights_h5file (str, optional): path to the HDF5 file containing
        feat_arr (np.ndarray): initialized nan-valued array of shape (n_domains, n_traces) to store the feature value.
        _: not used, just to match the function signature

    Returns:
        (np.ndarray): updated array of shape (n_domains, n_traces) with the feature values.
    """
    
    # Get the bandwidth sigma from the configuration
    sigma = config['sigma']
    
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
    
    # If the bedfile is provided, read it to only consider the domains in the BED file
    if 'bedfile' in config:
        
        # Check that the file exists
        if not os.path.isfile(config['bedfile']):
            raise ValueError(f"The BED file {config['bedfile']} does not exist.")
        
        # If a BED labels is provided, read it, otherwise assume it's a boolean 'True'
        try:
            bed_label = config['bed_label']
        except KeyError:
            bed_label = True
        
        # Get the BED values sorted by how spots appear in the cell data
        bedvals = cte.get_bed_values_by_spotIDs(cellID, config['bedfile'])
        
        # Get the mask to select spots with the given BED label
        mask = bedvals == bed_label
        
        # Filter the coordinates
        crds = crds[mask]
    
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

                # Calculate the distance of this spot to the other spots in 'crds'
                point = np.array([[x, y, z]])
                dists = cdist(point, crds)[0]
                
                # Remove the distance to the spot itself
                dists = dists[dists != 0]
                
                # If there are no distances, skip this spot
                if len(dists) == 0:
                    continue
                
                # Calculate the Gaussian Kernel Density for this spot
                feat_val = kernel_density(dists, sigma)
                
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
