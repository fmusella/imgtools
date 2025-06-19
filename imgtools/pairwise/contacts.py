import h5py
import numpy as np
from scipy.spatial import cKDTree
from alabtools.utils import map_indices
from ..cte import ChromatinTracingExperiment
from .pairwise_utils import *
from .. import parallel


# FUNCTIONS TO CALCULATE INTRA AND INTER CONTACT / COPRESENCE MATRICES

def calculate_intra_matrices(
    xs: np.ndarray, ys: np.ndarray, zs: np.ndarray,
    bins: np.ndarray, N: int, thresh: float, binarize: bool = False
) -> tuple:
    """ Calculate the co-presence and contact matrices.
    
    The co-presence matrix is a binary matrix that indicates whether
    two bins are co-present in the same cell. E.g.:
        bins_present = [0, 1, 4] (out of [0, 1, 2, 3, 4])
        cop_mat = [[0, 1, 0, 0, 1],
                   [1, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [1, 1, 0, 0, 0]]
    (the diagonal is set to 0 because it would be a trivial 1).
    
    The contact matrix is a binary matrix that indicates whether
    two bins are in contact. E.g.:
        bins_contact = [0, 4]
        ctc_mat = [[0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [1, 0, 0, 0, 0]]
    (also here the diagonal is set to 0).

    Args:
        xs (np.ndarray): x-coordinates of the spots
        ys (np.ndarray): y-coordinates of the spots
        zs (np.ndarray): z-coordinates of the spots
        bins (np.ndarray): bins of the spots
        N (int): number of bins in the chromosome
        thresh (float): threshold for 3D contact
        binarize (bool): whether to binarize the contact matrix.
            Defaults to False.

    Returns:
        cop_mat (np.ndarray): co-presence matrix
        ctc_mat (np.ndarray): contact matrix
    """
    
    # Create the co-presence matrix
    # Get all the bin pairs combinations
    i, j = np.triu_indices(bins.size, k=1)
    bins1, bins2 = bins[i], bins[j]
    # Initialize the co-presence matrix
    cop_mat = np.zeros((N, N), dtype=np.int32)
    np.add.at(cop_mat, (bins1, bins2), 1)
    np.add.at(cop_mat, (bins2, bins1), 1)  # make it symmetric
    
    # Calculate the pairwise contacts
    crds = np.column_stack((xs, ys, zs))  # shape (N, 3)
    # Create a KDTree for fast nearest neighbor search
    tree = cKDTree(crds)
    # Get the pairs in contact
    pairs = tree.query_pairs(thresh, output_type='ndarray')
    # Get the bins of the pairs
    bins1, bins2 = bins[pairs[:, 0]], bins[pairs[:, 1]]
    
    # Calculate the contact matrix
    ctc_mat = np.zeros((N, N), dtype=np.int32)
    np.add.at(ctc_mat, (bins1, bins2), 1)
    np.add.at(ctc_mat, (bins2, bins1), 1)  # make it symmetric
    
    # Make it binary if requested
    if binarize:
        cop_mat[cop_mat > 0] = 1
        ctc_mat[ctc_mat > 0] = 1
    
    # Remove the diagonal
    np.fill_diagonal(cop_mat, 0)
    np.fill_diagonal(ctc_mat, 0)
    
    return cop_mat, ctc_mat

def calculate_inter_matrices(
    xs_1: np.ndarray, ys_1: np.ndarray, zs_1: np.ndarray, bins_1: np.ndarray,
    xs_2: np.ndarray, ys_2: np.ndarray, zs_2: np.ndarray, bins_2: np.ndarray,
    N_1: int, N_2: int, thresh: float, binarize: bool = False
) -> tuple:
    """ Calculate the co-presence and contact matrices between spots from two different chromosomes.

    Args:
        xs_1 (np.ndarray): x coordinates of the spots from the first chromosome
        ys_1 (np.ndarray): y coordinates of the spots from the first chromosome
        zs_1 (np.ndarray): z coordinates of the spots from the first chromosome
        bins_1 (np.ndarray): Index-bins of the spots from the first chromosome
        xs_2 (np.ndarray): x coordinates of the spots from the second chromosome
        ys_2 (np.ndarray): y coordinates of the spots from the second chromosome
        zs_2 (np.ndarray): z coordinates of the spots from the second chromosome
        bins_2 (np.ndarray): Index-bins of the spots from the second chromosome
        N_1 (int): number of bins in the first chromosome
        N_2 (int): number of bins in the second chromosome
        thresh (float): 3D contact threshold
        binarize (bool): whether to binarize the contact matrix.
            Defaults to False.

    Returns:
        cop_mat (np.ndarray): co-presence matrix
        cnt_mat (np.ndarray): contact matrix
    """
    
    # Create the co-presence matrix
    # First, we count how many times each bin pair appears in the two arrays
    counts_1 = np.bincount(bins_1, minlength=N_1)
    counts_2 = np.bincount(bins_2, minlength=N_2)
    # Then, the co-presence matrix is the outer product of the counts
    cop_mat = np.outer(counts_1, counts_2)
    
    # Initialize the contact matrix
    cnt_mat = np.zeros((N_1, N_2), dtype=np.int32)
    
    # Calculate the pairwise contacts
    crds_1 = np.column_stack((xs_1, ys_1, zs_1))  # shape (N_1, 3)
    crds_2 = np.column_stack((xs_2, ys_2, zs_2))  # shape (N_2, 3)
    # Create a KDTree for fast nearest neighbor search
    tree_1 = cKDTree(crds_1)
    tree_2 = cKDTree(crds_2)
    # Get the pairs in contact between each point in crds_1 and each point in crds_2
    pairs = tree_1.query_ball_tree(tree_2, thresh)
    
    # Here pairs is a list of lists, where each list l[i] contains all the indices of crds_2
    # that are in contact with crds_1[i]. For example:
    #    pairs = [[], [11, 13], [0], [], [2, 4]]
    # means that idx 0 of crds_1 has no contacts, idx 1 has contacts with idx 11 and 13 of crds_2, etc.
    
    # We want convert them to numpy arrays of the form
    #   pairs_1 = [1, 1, 2, 4, 4], pairs_2 = [11, 13, 0, 2, 4]
    # where pairs_1[0], pairs_2[0] are the indices of the first contact, etc.
    
    # We first get the lengths of each list in pairs,
    # e.g. lens = [0, 2, 1, 0, 2] for the example above
    lens = np.fromiter((len(pair) for pair in pairs), dtype=int)
    
    # If there are contacts, we can proceed
    if lens.sum() > 0:
        
        # Repeat the indices of the first array for the number of contacts it has,
        # e.g. [1, 1, 2, 4, 4] for the example above
        # (note that a 0-length list will not be repeated, so it's absent)
        rows = np.repeat(np.arange(len(pairs)), lens)
        # Concatenate the pairs
        # e.g. [11, 13, 0, 2, 4] for the example above
        cols = np.concatenate(pairs).astype(int)
        
        # Add the contacts to the contact matrix
        np.add.at(cnt_mat, (bins_1[rows], bins_2[cols]), 1)
        
    # Make it binary if requested
    if binarize:
        cop_mat[cop_mat > 0] = 1
        cnt_mat[cnt_mat > 0] = 1
    
    return cop_mat, cnt_mat


# FUNCTIONS TO COLLECT AND UPDATE AVERAGE CONTACT / CO-PRESENCE / FREQUENCY MATRICES

def initialize_collector(n_1: int, n_2: int, cte: ChromatinTracingExperiment) -> dict:
    """ Initialize the collector dictionary to store the average matrices.
    
    For each (unique) state label in the CTE, the following keys are created:
        - 'nsamples': an integer value that counts the number of samples analyzed.
            The number of samples is equal to the number of cells times the number of homologous pairs,
            which should be 2 for intra-chromosomal contacts and 4 for inter-chromosomal contacts.
        - 'c_avg': a matrix of shape (n_1, n_2) to store the average number of contacts per bin pair.
        - 'c_var': a matrix of shape (n_1, n_2) to store the variance of contacts per bin pair.
        - 'n_avg': a matrix of shape (n_1, n_2) to store the average number of 'co-presence' per bin pair.
            This constitutes an upper limit for the number of contacts per bin pair,
        - 'n_var': a matrix of shape (n_1, n_2) to store the variance of 'co-presence' per bin pair.
    
    The 'all' state is also added to the list of states, which will contain the matrices
    for all cells regardless of their state.

    Args:
        n_1 (int): number of bins in the first chromosome
        n_2 (int): number of bins in the second chromosome
        cte (ChromatinTracingExperiment)

    Returns:
        (dict): a dictionary with the format:
            {state: {'nsamples': int, 'c_avg': np.ndarray, 'c_var': np.ndarray,
                     'n_avg': np.ndarray, 'n_var': np.ndarray}, ...}
    """
    
    # Get the unique states from the CTE
    states = np.unique(cte.cell_states)
    # Add the 'all' state to the list of states
    states = np.append(states, 'all')
    
    # Initialize the collector dictionary
    collector = {}
    
    # Initialize a null matrix for each state
    for state in states:
        collector[state] = {
            'nsamples': 0,
            'c_avg': np.zeros((n_1, n_2), dtype=np.float64),
            'c_var': np.zeros((n_1, n_2), dtype=np.float64),
            'n_avg': np.zeros((n_1, n_2), dtype=np.float64),
            'n_var': np.zeros((n_1, n_2), dtype=np.float64)
        }
    
    return collector

def update_collector(collector: dict, state: str, c: np.ndarray, n: np.ndarray) -> None:
    """ Update the matrices in the collector with the contact and co-presence matrices
    from the new cell (c and n).
    
    Uses the Welford's method to update the mean and std matrices to avoid numerical instability.
    
    Update the collector in-place.
    
    Args:
        collector (dict): collector dictionary to update
        state (str): state label for which to update the matrices
        c (np.ndarray): new contact matrix of shape (n_1, n_2)
        n (np.ndarray): new co-presence matrix of shape (n_1, n_2)
    """
    
    # If the co-presence matrix is empty, exit
    # We don't update the number of samples in this case.
    # There is clearly something wrong with the imaging,
    # so including these 0s is probably just wrong.
    if not n.any():
        return
    
    # Get the current matrices
    nsamples = collector[state]['nsamples']
    c_avg = collector[state]['c_avg']
    c_var = collector[state]['c_var']
    n_avg = collector[state]['n_avg']
    n_var = collector[state]['n_var']
    
    # Update the number of samples
    nsamples += 1
    
    # Update the contact matrix
    c_avg, c_var = welford_update_matrix(nsamples, c_avg, c_var, c)
    
    # Update the co-presence matrix
    n_avg, n_var = welford_update_matrix(nsamples, n_avg, n_var, n)
    
    # Update the group with the new matrices
    collector[state]['nsamples'] = nsamples
    collector[state]['c_avg'] = c_avg
    collector[state]['c_var'] = c_var
    collector[state]['n_avg'] = n_avg
    collector[state]['n_var'] = n_var

def collect_contact_frequency(collector: dict) -> None:
    """ Calculate the contact frequency and its variance for the matrices in each state,
    and save them as 'f' and 'f_var' in the collector dictionary.
    
    Update the collector in-place.

    Args:
        collector (dict): collector dictionary with the matrices
    """
    
    # Loop over the states in the collector
    for state in collector:
    
        # Get the matrices
        nsamples = collector[state]['nsamples']
        c_avg = collector[state]['c_avg']
        c_var = collector[state]['c_var']
        n_avg = collector[state]['n_avg']
        n_var = collector[state]['n_var']
        
        # Calculate the contact frequency
        f = c_avg / n_avg
        
        # Calculate the variance on f using the delta method
        f_var = (c_var - f ** 2 * n_var) / (nsamples * n_avg ** 2)
        
        # Save the contact frequency and its variance
        collector[state]['f'] = f
        collector[state]['f_var'] = f_var

def streamline_collector(collector: dict) -> None:
    """ Streamline the collector dictionary by removing unnecessary keys,
    i.e. 'c_avg', 'c_var', 'n_avg', 'n_var'.
    
    These keys are not needed anymore after calculating the contact frequency.
    
    After this function, the collector will only contain:
        - 'nsamples': int, # number of samples analyzed
        - 'f': np.ndarray, # average contact frequency matrix
        - 'f_var': np.ndarray, # variance of contact frequency matrix
    
    Update the collector in-place.

    Args:
        collector (dict)
    """
    
    for state in collector:
        del collector[state]['c_avg']
        del collector[state]['c_var']
        del collector[state]['n_avg']
        del collector[state]['n_var']


# FUNCTIONS TO PARALLELIZE THE CONTACT CALCULATION

def func_node(chrom_pair: tuple, cte_name: str, _, config: dict) -> dict:
    """ Node-level function to calculate the contact frequency (and all related matrices)
    between a pair of chromosomes across all cells in a Chromatin Tracing Experiment (CTE).
    
    This function performs the following steps:
    1. Reads the CTE file and its (high-resolution) index.
    2. Maps the CTE index to the target (low-resolution) index.
    3. Initializes a collector dictionary to store the average matrices for each state.
    4. Loops over all cells in the CTE.
    5. For each cell, it calculates the contact matrix and the co-presence matrix
       between all pairs of traceIDs of the two chromosomes.
    6. Updates the average and variance matrices in the collector for each state
       using Welford's method.
    7. Calculates the contact frequency and its variance for each state.
    
    The data is returned in the collector dictionary, which has the following format:
    collector[state] = {'nsamples': int, # number of samples analyzed,
                        'c_avg': np.ndarray, # average contact matrix
                        'c_var': np.ndarray, # variance of contact matrix
                        'n_avg': np.ndarray, # average co-presence matrix
                        'n_var': np.ndarray, # variance of co-presence matrix
                        'f': np.ndarray, # contact frequency matrix
                        'f_var': np.ndarray} # variance of contact frequency matrix

    Args:
        -: not used, just to match the signature of the function.
        chrom_pair (tuple): a tuple of two chromosome names (chrom_1, chrom_2)
        cte_name (str)
        config (dict): configuration dictionary containing:
            - 'resolution': for the target Index
                (used in the function read_target_index)
            - 'thresh': threshold for 3D contact
            - 'binarize': whether to binarize the contact matrix
            - 'filename': name of the HDF5 file to store the results
    
    Returns:
        collector (dict): a dictionary with the average matrices for each state. Format:
            {
                state: {
                    'nsamples': int, # number of samples analyzed
                    'c_avg': np.ndarray, # average contact matrix
                    'c_var': np.ndarray, # variance of contact matrix
                    'n_avg': np.ndarray, # average co-presence matrix
                    'n_var': np.ndarray, # variance of co-presence matrix
                    'f': np.ndarray, # contact frequency matrix
                    'f_var': np.ndarray, # variance of contact frequency matrix
                }
            }
    """
    
    # Get the chromosomes for the current pair
    chrom_1, chrom_2 = chrom_pair
    
    # Read the CTE file and its index
    cte = ChromatinTracingExperiment(cte_name, 'r')
    # Get the high-resolution Index from the CTE
    index_hres = cte.index
    
    # Read the target, low-resolution Index from the config
    index_lres = read_target_index(cte, config)
    # Get the target index hashmap
    index_lres_hashmap = index_lres.get_index_hashmap()
    # Get the offsets and the lengths of the chromosomes in the target Index
    ind_chrom_1 = np.where(index_lres.genome.chroms == chrom_1)[0][0]
    ind_chrom_2 = np.where(index_lres.genome.chroms == chrom_2)[0][0]
    # The offset gives the starting bin position of each chromosome in the Index.
    # For example, if index.chromstr = ['chr1', 'chr1', 'chr1', 'chr1', 'chr2', 'chr2', 'chr3', ...],
    # then offset_chrom_1 = 0, offset_chrom_2 = 4, offset_chrom_3 = 6, etc.
    offset_lres_chrom_1, offset_lres_chrom_2 = index_lres.offset[ind_chrom_1], index_lres.offset[ind_chrom_2]
    # The size gives the number of bins in each chromosome in the Index.
    # In the previous example, size_chrom_1 = 4, size_chrom_2 = 2, etc.
    size_lres_chrom_1, size_lres_chrom_2 = index_lres.chrom_sizes[ind_chrom_1], index_lres.chrom_sizes[ind_chrom_2]
    
    # Map the high-resolution domains to the low-resolution ones:
    #   {(chrom, start_hres, end_hres): [(chrom, start_lres, end_lres)], ...}
    domains_map_lres_to_hres = map_indices(index_hres, index_lres)
    
    # Initialize the collector dictionary to store the average matrices
    collector = initialize_collector(size_lres_chrom_1, size_lres_chrom_2, cte)
    
    # Loop over the cells
    for cellID, state in zip(cte.cell_labels, cte.cell_states):
        
        # Get the dict that maps chromosomes to their traceIDs in the cell: traceID_map[chrom] = [traceID_1, traceID_2]
        traceID_map = cte.get_trace_hashmap(cellID)
        
        # If either chromosome is not in the traceID_map, skip this pair
        if chrom_1 not in traceID_map or chrom_2 not in traceID_map:
            continue
        
        # Loop over the traceIDs of chromosome 1 in the cell
        for traceID_1 in traceID_map[chrom_1]:
            
            # Get the data of chrom_1 / traceID_1 in the cell
            xs_1, ys_1, zs_1, starts_hres_1, ends_hres_1, _, _ = cte.get_data(cellID, chrom_1, traceID_1, format='numpy')
            
            # Convert the domain info (chrom, start, end) of each spot
            # into its bin position along the low-resolution Index.
            bins_lres_1 = get_bins(chrom_1, starts_hres_1, ends_hres_1, domains_map_lres_to_hres, index_lres_hashmap)
            # Remove the offset of the chromosome, so that bins_1 start from 0
            bins_lres_1 = bins_lres_1 - offset_lres_chrom_1
            
            # If chrom_1 = chrom_2, this is an intra-chromosomal contact calculation
            # and we don't need to loop over the traceIDs of chrom_2.
            if chrom_1 == chrom_2:
                
                # Calculate the co-presence and contact matrices for the intra-chromosomal case
                cop_mat, cnt_mat = calculate_intra_matrices(
                    xs_1, ys_1, zs_1, bins_lres_1,
                    size_lres_chrom_1, config['thresh'], config['binarize']
                )
                
                # Update the collector with the new matrices
                update_collector(collector, state, cnt_mat, cop_mat)
                update_collector(collector, 'all', cnt_mat, cop_mat)
                
                continue
            
            # Otherwise, with chrom_1 != chrom_2, this is an inter-chromosomal contact calculation
            # Loop over the traceIDs of chromosome 2 in the cell
            for traceID_2 in traceID_map[chrom_2]:
                
                # Get the data of chrom_2 / traceID_2 in the cell
                xs_2, ys_2, zs_2, starts_hres_2, ends_hres_2, _, _ = cte.get_data(cellID, chrom_2, traceID_2, format='numpy')
                
                # Get the bins of chrom_2 as before
                bins_lres_2 = get_bins(chrom_2, starts_hres_2, ends_hres_2, domains_map_lres_to_hres, index_lres_hashmap)
                bins_lres_2 = bins_lres_2 - offset_lres_chrom_2  # Remove the offset of the chromosome
                
                # Calculate the co-presence and contact matrices
                cop_mat, cnt_mat = calculate_inter_matrices(
                    xs_1, ys_1, zs_1, bins_lres_1,
                    xs_2, ys_2, zs_2, bins_lres_2,
                    size_lres_chrom_1, size_lres_chrom_2,
                    config['thresh'], config['binarize']
                )
                
                # Update the collector with the new matrices
                update_collector(collector, state, cnt_mat, cop_mat)
                update_collector(collector, 'all', cnt_mat, cop_mat)
    
    # Calculate the contact frequency and its variance for each state
    # This will add the 'f' and 'f_var' keys to the collector dictionary in each state
    collect_contact_frequency(collector)
    
    # Streamline the collector by removing unnecessary keys, so we don't store unnecessary data
    # This will remove 'c_avg', 'c_var', 'n_avg', 'n_var' from each state,
    # leaving only 'nsamples', 'f', and 'f_var'.
    streamline_collector(collector)
    
    return collector

def reduce_initialization(_1, cte_name: str, _2, config: dict) -> None:
    """ Initialize the HDF5 file for storing the final matrices.
    
    Creates:
        - the target, low-resolution Index,
        - a group for each state in the CTE,
        - an 'all' group that contains the matrices for all cells
          regardless of their state,
        - within each group, two sub-groups: 'intra' and 'inter'.
    
    Since we create the HDF5 file to collect the nodes' results,
    we don't have to return anything from this function.

    Args:
        _*: not used, just to match the signature of the function.
        cte_name (str)
        config (dict): configuration dictionary containing:
            - 'filename': name of the HDF5 file to store the results
    """
    
    # Get the unique states from the CTE
    cte = ChromatinTracingExperiment(cte_name, 'r')
    states = np.unique(cte.cell_states)
    # Add the 'all' state to the list of states
    states = np.append(states, 'all')
    
    # Open the HDF5 file for writing
    h5 = h5py.File(config['filename'], 'w')
    
    # Save the low-resolution, target Index in the HDF5 file
    index_lres = read_target_index(cte, config)
    index_lres.save(h5)
    
    # Create a group for each state
    for state in states:
        state_group = h5.create_group(state)
        
        # Create an intra and an inter group
        state_group.create_group('intra')
        state_group.create_group('inter')
    
    h5.close()

def reduce_update(chrom_pair: tuple, _1, pair_collector: dict, _2, _3, config: dict) -> None:
    """ Update the HDF5 file with the results of the pairwise calculations.
    
    Since we are collecting the results from the nodes in the HDF5 file,
    there isn't a general collector dictionary to update.

    Args:
        _*: not used, just to match the signature of the function.
        chrom_pair (tuple): a tuple of two chromosome names (chrom_1, chrom_2)
        pair_collector (dict): collector dictionary for the current chromosome pair,
            containing the average matrices for each state.
        config (dict): configuration dictionary containing:
            - 'filename': name of the HDF5 file to store the results
    """
    
    # Get the chromosomes for the current pair
    chrom_1, chrom_2 = chrom_pair
    
    # Get the states from the collector
    states = list(pair_collector.keys())
    
    # Read the h5 file for the collected matrices
    try:
        h5 = h5py.File(config['filename'], 'a')
    except OSError as e:
        raise OSError(f"Error opening HDF5 file {config['filename']}: {e}")

    # Loop over the states
    for state in states:
        
        # Create a group for the chromosome pair in the HDF5 file
        # If the chromosome pair is intra-chromosomal, we use the 'intra' group
        if chrom_1 == chrom_2:
            pair_group = h5[state]['intra'].create_group(chrom_1)
        # Otherwise, we use the 'inter' group
        else:
            pair_group = h5[state]['inter'].create_group(f'{chrom_1}_{chrom_2}')
        
        # Store the matrices in the group
        pair_group.create_dataset('nsamples', data=pair_collector[state]['nsamples'], dtype=np.int64)
        pair_group.create_dataset('f', data=pair_collector[state]['f'], dtype=np.float64, chunks=True, compression='gzip')
        pair_group.create_dataset('f_var', data=pair_collector[state]['f_var'], dtype=np.float64, chunks=True, compression='gzip')
    
    h5.close()


# MAIN FUNCTION TO RUN THE CONTACT CALCULATION

# Define the required keys for the configuration dictionary
required_keys = {
    'resolution': {'type': [str, int]},
    'thresh': {'type': [int, float], 'positive': True},
    'binarize': {'type': bool},
    'filename': {'type': str}
}

def main(cte: ChromatinTracingExperiment, config: dict) -> None:
    
    parallel.control_func(
        cte, None,
        config, required_keys,
        func_node, reduce_initialization, reduce_update,
        mode = 'chrom_pair'
    )
