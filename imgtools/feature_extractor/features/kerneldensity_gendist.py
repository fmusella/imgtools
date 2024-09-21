import os
import h5py
import numpy as np
from scipy.spatial.distance import cdist
from alabtools.utils import Index, get_index_from_bed
from ...cte import ChromatinTracingExperiment
from ...cte import cte_utils

docstring = """"""

required_keys = {
    'sigma': {'type': float, 'positive': True},
}


def read_bedfile(bedfile: str, index: Index) -> Index:
    """ Read the BED file with the list of domains to consider for the Kernel Density.
    
    The BED file can be in 'collapsed' or 'expanded' format.
    
    In the 'collapsed' format, the BED file has only 3 columns: chrom, start, end.
    The domains are the regions defined by the start and end coordinates.
    
    In the 'expanded' format, the BED file has 4 columns: chrom, start, end, track0.
    All the regions of the original Index are present, and the fourth column is a boolean
    that indicates whether the region is a domain or not.
    
    This function reads the BED file and checks the format.
    If the BED file is in 'collapsed' format, it returns the Index as is.
    If the BED file is in 'expanded' format, it filters the Index to keep only the domains,
    creating a 'collapsed' Index with only the domains selected.

    Args:
        bedfile (str): path to the BED file.
        index (Index): the original Index of the CTE.

    Returns:
        Index: the collapsed BED Index with only the domains selected.
    """
    
    bed = get_index_from_bed(bedfile, genome=index.genome)
    
    # Check that the bed Index has at most 4 columns,
    # i.e. that there is no 'track1' in the attributes
    if hasattr(bed, 'track1'):
        raise ValueError(f"Error: the BED file {bedfile} should have at most 4 columns.")
    
    # If the bed only has 3 columns, these are the chrom, start and end,
    # and it means that it is in the 'collapsed' format. Just return it.
    if not hasattr(bed, 'track0'):
        return bed
    
    # If the bed has 4 columns, it means that it is in the 'expanded' format.
    # First let's check that the bed Index matches the original Index.
    if bed != index:
        raise ValueError(f"Error: the BED file {bedfile} does not match the CTE Index.")
    # The fourth column should be boolean, and it should be True for the domains
    domains = bed.track0
    try:  # Try to convert the fourth column to boolean
        domains = np.array(domains, dtype=bool)
    except ValueError:
        raise ValueError(f"Error: the fourth column of the BED file {bedfile} should be boolean.")
    
    # Filter the bed Index to keep only the domains
    chromstr = bed.chromstr[domains]
    start = bed.start[domains]
    end = bed.end[domains]
    
    # Return the collapsed BED Index
    return Index(chromstr, start, end, genome=index.genome)

def kernel_density(dists: np.ndarray, sigma: float, weights: np.array = None) -> float:
    """ Calculate the Kernel Density for a set of distances:
        K = (1 / N) * (1 / ((2 * pi)^1.5 * sigma^3)) * sum_j [w_j * e^(-d_j^2 / (2 * sigma^2))],
    where:
        - K is the Kernel Density,
        - N is the total number of distances,
        - d_j is the j-th distance (corresponding to ||x_i - x_j||),
        - sigma is the bandwidth of the Gaussian Kernel Density,
        - w_j is the j-th weight (1 if not provided).

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
    """ 
    """
    
    # Get the bandwidth sigma from the configuration
    sigma = config['sigma']
    
    # Get the cell data in dictionary format
    cell_data = cte.get_data(cellID)
    
    # Get the traceID hash table to map traces to their position in the array
    traceID_hash = cte.get_trace_hashmap(cellID)
    
    # Convert the cell data in numpy format and get the genomic coordinates of the spots
    _, _, _, chroms, starts, ends, _, _, _ = cte_utils.cell_dict_to_numpy(cell_data)
    
    # Get the index and its hash table
    index = cte.index
    index_hash = index.get_index_hashmap()
    
    # If the weights file is provided as a HDF5 file, read the weights for this cell
    if 'weights_h5file' in config:
        
        # Check that the file exists
        if not os.path.isfile(config['weights_h5file']):
            raise ValueError(f"The weights HDF5 file {config['weights_h5file']} does not exist.")
        
        # Open the HDF5 file
        try:
            h5 = h5py.File(config['weights_h5file'], 'r')
        except Exception as e:
            raise ValueError(f"Error opening the weights file as HDF5 file: {e}")
       
        # Read the weights for this cell
        try:
            weights = h5[cellID][:]
        except Exception as e:
            h5.close()
            raise ValueError(f"Error reading the weights for cell {cellID}: {e}")
        
        # Check that the shape of the weights matches the data
        if not weights.shape == chroms.shape:
            h5.close()
            raise ValueError(f"Error: shape of weights ({weights.shape}) does not match the data ({xs.shape}).")
        h5.close()
    
    # Otherwise, set the weights to None
    else:
        weights = None
    
    # If the bedfile is provided, read it to only consider the domains in the BED file
    if 'domain_bedfile' in config:
        
        # Check that the file exists
        if not os.path.isfile(config['domain_bedfile']):
            raise ValueError(f"The BED file {config['domain_bedfile']} does not exist.")
        
        # Read the BED file and get the collapsed Index
        bed = read_bedfile(config['domain_bedfile'], index)
        bed_hash = bed.get_index_hashmap()  # Get the hash table of the BED Index
        
        # Get a mask to filter the coordinates of the domains of interest
        mask = []
        for chrom, start, end in zip(chroms, starts, ends):
            if (chrom, start, end) in bed_hash:
                mask.append(True)
            else:
                mask.append(False)
        mask = np.array(mask).astype(bool)
        
        # Filter the genomic coordinates (and the weights if provided)
        chroms = chroms[mask]
        starts = starts[mask]
        ends = ends[mask]
        if weights is not None:
            weights = weights[mask]
    
    # Initialize a dictionary to store the feature values for each domain (we will then take the average)
    feat_per_domain = {}
    
    for chrom in cell_data:       
        for traceID in cell_data[chrom]:
            
            # Get the position of the trace in the array
            i_trace = traceID_hash[chrom][traceID]
            
            for spotID in cell_data[chrom][traceID]:
                
                # Unpack the spot data
                spot_data = cell_data[chrom][traceID][spotID]
                start, end = spot_data['start'], spot_data['end']

                # Calculate the genomic distances between the spot and all the other spots
                # (we only consider the start coordinates for simplicity)
                start_dists = np.abs(starts - start)
                chrom_dists = chroms == chrom
                
                # Remove distances equal to 0 (the spot itself)
                # and on different chromosomes (irrelevant)
                mask = np.logical_and(start_dists != 0, chrom_dists)
                dists = start_dists[mask]
                if weights is not None:
                    # Create another variable, otherwise weights would be modified for the entire loop
                    weights_ = weights[mask]
                else:
                    weights_ = None
                
                # If there are no distances, skip this spot
                if len(dists) == 0:
                    continue
                
                # Calculate the Gaussian Kernel Density for this spot
                feat_val = kernel_density(dists, sigma, weights_)
                
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
