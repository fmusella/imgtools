import os
import h5py
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from scipy.spatial.distance import squareform
from scipy.sparse.linalg import eigsh
from statsmodels.stats.multitest import multipletests
from alabtools.utils import Index, map_indices
from ..cte import ChromatinTracingExperiment
from .pairwise_utils import *
from .. import parallel
from ..utils import clean_correlation


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
    
    If there are no states, only the 'all' state will be initialized.

    Args:
        n_1 (int): number of bins in the first chromosome
        n_2 (int): number of bins in the second chromosome
        cte (ChromatinTracingExperiment)

    Returns:
        (dict): a dictionary with the format:
            {state: {'nsamples': int, 'c_avg': np.ndarray, 'c_var': np.ndarray,
                     'n_avg': np.ndarray, 'n_var': np.ndarray}, ...}
    """
    
    # Get the unique states from the CTE, if available
    if 'cell_states' in cte:
        states = np.unique(cte.cell_states)
    # Otherwise, create an empty array
    else:
        states = np.array([], dtype=str)
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
    
    MATH:
    
    The formula for the contact frequency is f = c* / n* (using * instead of avg for simplicity).
    
    The variance, propagated using the delta method, is:
        Var(f) = 1 / n*^2 * Var(c*) + (c*^2 / n*^4) * Var(n*) - 2 (c* / n*^3) * cov(c*, n*).
    
    The covariance term can be expanded as:
        cov(c*, n*) = cov(sum_i c_i / T, sum_j n_j / T) = 1 / T^2 * sum_i sum_j cov(c_i, n_j)
    where T is the number of samples (nsamples).
    Since i and j indicate cells, we can assume that cov(c_i, n_j) = 0 for i != j, because they are independent samples.
    Also we can assume that cov(c_i, n_i) = cov(c, n) for all i, because they are identically distributed samples.
    So the covariance term can be simplified to:
        cov(c*, n*) = 1 / T^2 * sum_i cov(c_i, n_i) = 1 / T^2 * T * cov(c, n) = 1 / T * cov(c, n)
    
    Using the covariance formula:
        cov(c, n) = E[c * n] - E[c] * E[n].
    To estimate E[c * n] - that now we indicate as E[C * N] to stress that C and N are random variables -
    we simply calculate the expectation:
        E[C * N] = sum_n=0_to_max(n) sum_c=0_to_n [c * n * P(C=c, N=n)] = 
                 = sum_n=0_to_max(n) P(N=n) * sum_c=0_to_n [c * P(C=c | N=n)].
    Reasons for the summations: the sum of N goes up to max(n) because in general we count the number of high-resolution co-presence pairs
    in the low-resolution bin pairs. If the resolutions match, max(n) = 1 obviously.
    Then, the observed contacts are obviously limited by the observed co-presence, so 0 <= c <= n.
    Finally, we assume that P(C=c | N=n) follows a binomial distribution: given n co-presence pairs, the number of observed contacts c
    follows a binomial distirbution with n trials and success probability f - the estimated contact frequency.
    With this assumption, we can calculate the inner sum as:
        sum_c=0_to_n [c * P(C=c | N=n)] = n * f - since it's just the expectation of a binomial distribution with n trials and success probability f.
    So the formula for E[C * N] can be simplified to:
        E[C * N] = sum_n=0_to_max(n) [n * f * P(N=n)] = f * sum_n=0_to_max(n) [n^2 * P(N=n)] = f * E[N^2]
    
    We can then estimate everything:
        cov(c, n) = E[C * N] - E[C] * E[N] = f * (E[N^2] - E[N]^2) = f * Var(N).
        cov(c*, n*) = 1 / T * cov(c, n) = f * Var(n) / T = f * Var(n*).
        Var(f) = (1 / n*^2) * (Var(c*) - f^2 * Var(n*)).
    
    Since the collector measured Var(c) and Var(n) - and not Var(c*) and Var(n*) - we need to divide them by T (nsamples) to get the f variance:
        Var(f) = (1 / T * n*^2) * (Var(c) - f^2 * Var(n)),
    where T is the number of samples (nsamples).

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
    
    If the 'store_single_cell' option is set in the config,
    the function will also store the single-cell contact matrix in the temporary directory,
    where missing data is set to -1.

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
            - 'store_single_cell' (optional): whether to store the single-cell data.
                    If this key is missing, it's assumed to be False.
    
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
    
    # If 'store_single_cell' is True, we will also store the single-cell data
    if 'store_single_cell' in config:
        # Create an HDF5 file to store the single-cell data in the temporary directory
        sc_h5_name = os.path.join(config['tempdir'], f'{chrom_1}_{chrom_2}.single-cell.h5')
        sc_h5 = h5py.File(sc_h5_name, 'w')
    # Otherwise, we just set sc_h5 to None
    else:
        sc_h5 = None
    
    # Loop over the cells
    for i, cellID in enumerate(cte.cell_labels):
        # Get the state of the cell, if available
        if 'cell_states' in cte:
            state = cte.cell_states[i]
        # Otherwise, set state to None
        else:
            state = None
        
        # Get the dict that maps chromosomes to their traceIDs in the cell: traceID_map[chrom] = [traceID_1, traceID_2]
        traceID_map = cte.get_trace_hashmap(cellID)
        
        # If either chromosome is not in the traceID_map, skip this pair
        if chrom_1 not in traceID_map or chrom_2 not in traceID_map:
            continue
        
        # Loop over the traceIDs of chromosome 1 in the cell
        for traceID_1 in traceID_map[chrom_1]:
            
            # Get the data of chrom_1 / traceID_1 in the cell
            d = cte.get_data(cellID, chrom_1, traceID_1, format='numpy')
            xs_1, ys_1, zs_1, starts_hres_1, ends_hres_1 = d['xs'], d['ys'], d['zs'], d['starts'], d['ends']
            
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
                update_collector(collector, 'all', cnt_mat, cop_mat)
                if state is not None:
                    update_collector(collector, state, cnt_mat, cop_mat)
                
                # Store the single-cell data if requested
                if sc_h5 is not None:
                    # Set as -1 the entries in cnt_mat that are 0 in the cop_mat (i.e. missing contacts)
                    cnt_mat[cop_mat == 0] = -1
                    # Require a group for the cellID / intra contacts
                    cell_group = sc_h5.require_group(f'{cellID}')
                    # Create a dataset for the traceID
                    cell_group.create_dataset(traceID_1, data=cnt_mat, dtype=np.int32, chunks=True, compression='gzip')
                
                continue
            
            # Otherwise, with chrom_1 != chrom_2, this is an inter-chromosomal contact calculation
            # Loop over the traceIDs of chromosome 2 in the cell
            for traceID_2 in traceID_map[chrom_2]:
                
                # Get the data of chrom_2 / traceID_2 in the cell
                d = cte.get_data(cellID, chrom_2, traceID_2, format='numpy')
                xs_2, ys_2, zs_2, starts_hres_2, ends_hres_2 = d['xs'], d['ys'], d['zs'], d['starts'], d['ends']
                
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
                update_collector(collector, 'all', cnt_mat, cop_mat)
                if state is not None:
                    update_collector(collector, state, cnt_mat, cop_mat)
                
                # Store the single-cell data if requested
                if sc_h5 is not None:
                    # Set as -1 the entries in cnt_mat that are 0 in the cop_mat (i.e. missing contacts)
                    cnt_mat[cop_mat == 0] = -1
                    # Require a group for the cellID / inter contacts
                    cell_group = sc_h5.require_group(f'{cellID}')
                    # Create a dataset for the traceID pair
                    cell_group.create_dataset(f'{traceID_1}_{traceID_2}', data=cnt_mat, dtype=np.int32, chunks=True, compression='gzip')
    
    # Calculate the contact frequency and its variance for each state
    # This will add the 'f' and 'f_var' keys to the collector dictionary in each state
    collect_contact_frequency(collector)
    
    # Streamline the collector by removing unnecessary keys, so we don't store unnecessary data
    # This will remove 'c_avg', 'c_var', 'n_avg', 'n_var' from each state,
    # leaving only 'nsamples', 'f', and 'f_var'.
    streamline_collector(collector)
    
    # Close the single-cell HDF5 file if it was created
    if sc_h5 is not None:
        sc_h5.close()
    
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
    
    If the 'store_single_cell' option is set in the config,
    we also create an HDF5 file for the single-cell data,
    with the following structure:
        - the target, low-resolution Index,
        - the cell labels,
        - the cell states,
        - a group 'contact_maps'.
    The name of the single-cell HDF5 file is the same as the main one,
    but with '.single-cell.h5' instead of '.h5'.

    Args:
        _*: not used, just to match the signature of the function.
        cte_name (str)
        config (dict): configuration dictionary containing:
            - 'filename': name of the HDF5 file to store the results
    """
    
    # Get the unique states from the CTE, if available
    cte = ChromatinTracingExperiment(cte_name, 'r')
    if 'cell_states' in cte:
        states = np.unique(cte.cell_states)
    # Otherwise, create an empty array
    else:
        states = np.array([], dtype=str)
    # Add the 'all' state to the list of states
    states = np.append(states, 'all')
    
    # Get the target, low-resolution Index from the CTE
    index_lres = read_target_index(cte, config)
    
    # Open the HDF5 file for writing the average matrices
    with h5py.File(config['filename'], 'w') as h5:
    
        # Save the low-resolution, target Index in the HDF5 file
        index_lres.save(h5)
        # Save the unique states
        h5.create_dataset('states', data=states.astype('S'))
        # Create a group for each state
        for state in states:
            state_group = h5.create_group(state)
            # Create an intra and an inter group
            state_group.create_group('intra')
            state_group.create_group('inter')
    
    # If we don't store single-cell data, we are done
    if 'store_single_cell' not in config or config['store_single_cell'] is False:
        return None
    
    # Otherwise, we also need to create the HDF5 file for the single-cell data.
    sc_h5_name = config['filename'].replace('.h5', '.single-cell.h5')
    with h5py.File(sc_h5_name, 'w') as sc_h5:
        
        # Save the low-resolution, target Index
        index_lres.save(sc_h5)
        # Save the cellIDs
        sc_h5.create_dataset('cell_labels', data=cte.cell_labels.astype('S'))
        # Save the cell states, if available
        if 'cell_states' in cte:
            sc_h5.create_dataset('cell_states', data=cte.cell_states.astype('S'))
        else:
            pass
        # Create a group for the contact maps
        sc_h5.create_group('contact_maps')

def reduce_update(chrom_pair: tuple, _1, pair_collector: dict, _2, _3, config: dict) -> None:
    """ Update the HDF5 file with the results of the pairwise calculations.
    
    Since we are collecting the results from the nodes in the HDF5 file,
    there isn't a general collector dictionary to update.
    
    If the 'store_single_cell' option is set in the config,
    we also update the single-cell HDF5 file with the single-cell contact matrices
    for the current chromosome pair.

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
    with h5py.File(config['filename'], 'a') as h5:

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
    
    # If we don't store single-cell data, we are done
    if 'store_single_cell' not in config or config['store_single_cell'] is False:
        return None
        
    # Otherwise, we also need to update the single-cell data.
    sc_h5_name = config['filename'].replace('.h5', '.single-cell.h5')
    with h5py.File(sc_h5_name, 'a') as sc_h5:
    
        # Read the chrom_pair h5 file from the temporary directory
        sc_pair_h5_name = os.path.join(config['tempdir'], f'{chrom_1}_{chrom_2}.single-cell.h5')
        try:
            sc_pair_h5 = h5py.File(sc_pair_h5_name, 'r')
        except OSError as e:
            raise OSError(f"Error opening HDF5 file {sc_pair_h5_name}: {e}")
        
        # Loop over the cells in the pair h5 file
        for cellID in sc_pair_h5.keys():
            
            # Require the cell group in the global single-cell h5 file
            cell_group = sc_h5.require_group(f'contact_maps/{cellID}')
            # Require a group for the chromosome pair
            if chrom_1 == chrom_2:
                pair_group = cell_group.require_group(f'intra/{chrom_1}')
            else:
                pair_group = cell_group.require_group(f'inter/{chrom_1}_{chrom_2}')
            
            # Loop over the traceIDs in the pair h5 file
            for trace_pair in sc_pair_h5[cellID].keys():
                
                # Get the contact matrix for the traceID
                cnt_mat = sc_pair_h5[cellID][trace_pair][:]
                # Create a dataset for the traceID in the single-cell h5 file
                pair_group.create_dataset(trace_pair, data=cnt_mat, dtype=np.int32, chunks=True, compression='gzip')


# MAIN FUNCTION TO RUN THE CONTACT CALCULATION

# Define the required keys for the configuration dictionary
required_keys = {
    'resolution': {'type': [str, int]},
    'thresh': {'type': [int, float], 'positive': True},
    'binarize': {'type': bool},
    'filename': {'type': str},
    'store_single_cell': {'type': bool, 'optional': True}
}

def run_contacts(cte: ChromatinTracingExperiment, config: dict) -> None:
    """ Runs the contact calculation for a Chromatin Tracing Experiment (CTE).
    
    The contacts are first calculated at the resolution of the CTE data (e.g. 100kb),
    and then mapped to the target resolution separately in each cell (e.g. 1Mb).
    
    This mapping can be performed in two ways:
        a) 'binarizing' the contacts in each cell. For example, we say that each 1Mb x 1Mb
           block is in contact if at least one 100kb x 100kb block in it is in contact.
        b) 'not binarizing' the contacts in each cell. In this case, we count the number
           of 100kb x 100kb blocks in contact in each 1Mb x 1Mb block.

    When calculating the contact frequency map, we average the number of contacts by
    the time each pair of bins is simultaneously imaged in the same cell.
    For the 'not binarizing' option, this means that we average the number of contacts
    by the number of 100kb x 100kb blocks simultaneously imaged in the same 1Mb x 1Mb block.
    
    We also calculate the variance on the mean contact frequency map using the delta method.
    Important: this variance is already divided by the number of samples.
    
    The calculation is performed separately for all the unique states in the CTE,
    including an 'all' state that averages across all cells.
    
    If the CTE doesn't have states, only the 'all' state is considered.
    
    The data is stored in an HDF5 file, with the following structure:
      - the target, low-resolution Index,
      - a group for each state (e.g. 'G1')
      - a sub-group for 'intra' or 'inter' contacts
      - a sub-group for each chromosome pair (e.g. 'chr1' for intra, 'chr1_chr2' for inter),
        with the following datasets:
        - 'nsamples' (int): the number of samples analyzed for this state,
            which provides the statistics for the contact frequency.
        - 'f' (np.ndarray): the average contact frequency matrix for this state.
        - 'f_var' (np.ndarray): the variance of the contact frequency matrix for this state,
            already divided by the number of samples.
      - at the root level, we also store the target, low-resolution Index used for the calculation.
    
    If the 'store_single_cell' option is set in the config, we also store the single-cell contact matrices.
    They are stored in a separate HDF5 file with the following structure:
      - the target, low-resolution Index,
      - the cell labels (as a dataset of strings),
      - the cell states (as a dataset of strings),
      - a group 'contact_maps' that contains the contact matrices for each cell:
        - a sub-group for each cellID,
        - within each cellID sub-group, two sub-groups: 'intra' and 'inter',
        - within each 'intra' or 'inter' sub-group, a sub-group for each chromosome / chromosome pair,
        - within each chromosome / chromosome pair sub-group, a dataset for each traceID / traceID pair.
          This dataset contains the contact matrix for that traceID / traceID pair.
    Note that, in the single-cell contact matrices, missing contacts are set to -1.

    Args:
        cte (ChromatinTracingExperiment)
        config (dict): configuration dictionary specifying the parameters for the calculation:
            - 'resolution': the target resolution to call the contacts at.
            - 'thresh': the threshold for 3D contact.
            - 'binarize': whether to binarize the contact matrix in each cell.
            - 'filename': the name of the HDF5 file to store the results.
            - 'store_single_cell' (optional): whether to store the single-cell contact matrices.
    """
    
    parallel.control_func(
        cte, None,
        config, required_keys,
        func_node, reduce_initialization, reduce_update,
        mode = 'chrom_pair'
    )


# FUNCTION TO RESCALE THE CONTACTS BETWEEN STATES

def rescale_contacts(h5_in: h5py.File, h5_out_name: str) -> h5py.File:
    """ Rescale the contact frequencies in the contacts H5 file,
    so that the average contact frequencies for each state (e.g. G1, S, G2)
    match the average contact frequencies of the 'all' state.
    
    This is done separately for intra- and inter-chromosomal contacts.

    Args:
        h5_in (h5py.File): input H5 file with the contact frequencies to rescale.
        h5_out_name (str): name of the output H5 file to store the rescaled contact frequencies.

    Returns:
        h5py.File: output H5 file with the rescaled contact frequencies.
    """
    
    # --- GET THE STATES AND CHECK THE INPUT H5 FILE ---
    
    # Get the states from the H5 file
    states = h5_in['states'][:].astype(str)
    # Ensure that 'all' is in the states
    if 'all' not in states:
        raise ValueError("The input H5 file must contain the 'all' state for rescaling.")
    # If there are no other states, nothing to rescale
    if len(states) == 1:
        raise ValueError("The input H5 file must contain at least one state other than 'all' for rescaling.")
    
    # --- CALCULATE THE RESCALING RATIOS USING THE 'ALL' STATE AS REFERENCE ---
    
    # Initialize two dictionaries to store the average contact frequencies and total counts
    # for each state and each contact type (intra and inter)
    avg_f = {state: {'intra': 0., 'inter': 0.} for state in states}
    ntotal = {state: {'intra': 0, 'inter': 0} for state in states}

    # Calculate the average intra contact frequencies
    for state in states:
        for chrom in h5_in[state]['intra']:
            
            # Get the intra chromosome contact frequency
            f = h5_in[state]['intra'][chrom]['f'][:]  # shape: (n, n)
            # Since it's a symmetric matrix, we can take the upper triangle
            f = squareform(f, checks=False)  # shape: (n * (n - 1) // 2,)
            # Remove NaN values
            f = f[~np.isnan(f)]
            
            # Update the average intra frequency and total count
            avg_f[state]['intra'] = (avg_f[state]['intra'] * ntotal[state]['intra'] + np.sum(f)) / (ntotal[state]['intra'] + len(f))
            ntotal[state]['intra'] += len(f)

    # Calculate the average inter contact frequencies
    for state in states:
        for chrom_pair in h5_in[state]['inter']:
            
            # Get the inter chromosome contact frequency
            f = h5_in[state]['inter'][chrom_pair]['f'][:]  # shape: (n1, n2)
            # Remove NaN values and flatten
            f = f[~np.isnan(f)]  # shape: (n1 * n2,)
            
            # Update the average inter frequency and total count
            avg_f[state]['inter'] = (avg_f[state]['inter'] * ntotal[state]['inter'] + np.sum(f)) / (ntotal[state]['inter'] + len(f))
            ntotal[state]['inter'] += len(f)

    # Calculate the ratio for rescaling, using 'all' as reference
    ratio = {}
    for state in states:
        ratio[state] = {}
        for contact_type in ['intra', 'inter']:
            ratio[state][contact_type] = avg_f['all'][contact_type] / avg_f[state][contact_type]

    # Print the averages and the ratios
    for state in states:
        print(f'State: {state}')
        for contact_type in ['intra', 'inter']:
            print(f'  Contact Type: {contact_type}')
            print(f'    Average Frequency: {avg_f[state][contact_type]:.4f}')
            print(f'    Rescale Ratio: {ratio[state][contact_type]:.4f}')

    # --- CREATE A NEW H5 FILE WITH RESCALED CONTACT FREQUENCIES ---

    # Create the output H5 file
    h5_out = h5py.File(h5_out_name, 'w')

    # Save the Index in the output H5 file
    index = Index(h5_in)
    index.save(h5_out)
    
    # Save the states in the output H5 file
    h5_out.create_dataset('states', data=states.astype('S'))

    # Loop over the states, and save the rescaled contact frequencies
    for state in states:
        
        # Create a group for the state
        state_group = h5_out.create_group(state)
        
        # Loop over the intra and inter contacts
        for contact_type in ['intra', 'inter']:
            # Create a group for the contact type
            contact_group = state_group.create_group(contact_type)
            
            # Loop over the chromosomes or chromosome pairs
            for chrom in h5_in[state][contact_type]:
                # Create a group for the chromosome or chromosome pair
                chrom_group = contact_group.create_group(chrom)
                
                # Get the contact frequency, variance, nsamples
                f = h5_in[state][contact_type][chrom]['f'][:]
                f_var = h5_in[state][contact_type][chrom]['f_var'][:]
                nsamples = int(h5_in[state][contact_type][chrom]['nsamples'][...])
                
                # Rescale the contact frequency and variance
                r = ratio[state][contact_type]
                f = f * r
                f_var = f_var * (r ** 2)
                
                # Save the rescaled contact frequency, variance, and nsamples
                chrom_group.create_dataset('nsamples', data=nsamples, dtype=np.int64)
                chrom_group.create_dataset('f', data=f, dtype=np.float64, chunks=True, compression='gzip')
                chrom_group.create_dataset('f_var', data=f_var, dtype=np.float64, chunks=True, compression='gzip')
    
    return h5_out


# FUNCTION TO IDENTIFY SIGNIFICANT CONTACTS

def call_significant_contacts(
    h5: h5py.File,
    alpha: float = 0.05,
    eff_size_intra: float = np.log2(1.5),
    eff_size_inter: float = np.log2(1.5)
) -> None:
    """ Identify significant contacts in the contact frequency matrices in each state.
    
    This function perform a z-test between each i-j entry to the control value, defined as:
       - 'inter': the average contact frequency across all inter contacts in the state,
       - 'intra': the average contact frequency at the same genomic distance.
    
    The z-test is one-sided, meaning that we only care about contacts that are significantly
    higher than the control value.
    
    The p-values are corrected for multiple testing using the Benjamini-Hochberg procedure,
    separately for intra and inter contacts and separately for each state.
    
    Furthermore, we only consider contacts as significant if their effect size (in log2 scale)
    is above specified thresholds.

    This function saves the significant contacts in the H5 file,
    in a new dataset 'significant' in each chromosome / chromosome pair group.
    
    Furthermore, the function saves a dataset 'control' in each chromosome / chromosome pair group,
    containing the control values used for the z-test.

    Args:
        h5 (h5py.File): H5 file with the contact frequency matrices.
        alpha (float, optional): Significance level for the corrected p-values of the z-test. Defaults to 0.05.
        eff_size_thresh_intra (float, optional): Minimum effect size (in log2 scale) for intra contacts. Defaults to log2(1.5).
        eff_size_thresh_inter (float, optional): Minimum effect size (in log2 scale) for inter contacts. Defaults to log2(1.5).
    """
    
    # Read the Index and the states from the H5 file
    index = Index(h5)
    states = h5['states'][:].astype(str)

    # --- READ AND FLATTEN THE MATRICES OF ALL CHROMOSOME PAIRS ---

    # Initialize the dictionaries to store the flattened arrays:
    #   - f_dict: flattened contact frequency
    #   - f_var_dict: flattened contact frequency variance
    #   - n_dict: flattened nsamples
    #   - pair_idx_dict: flattened chromosome pair indices
    #   - gendist_dict: flattened genomic distances (for intra only)
    f_dict, f_var_dict, n_dict, pair_idx_dict, gendist_dict = {}, {}, {}, {}, {}
    # We also store the shapes of the original matrices, so we can reshape later
    shape_dict = {}

    # Loop through the states
    for state in states:
        
        # Initialize dictionaries for the current state
        f_dict[state] = {}
        f_var_dict[state] = {}
        n_dict[state] = {}
        pair_idx_dict[state] = {}
        gendist_dict[state] = []  # only for intra, so directly a list
        shape_dict[state] = {}
        
        # Loop through the contact types
        for contact_type in ['intra', 'inter']:
            
            # Initialize the lists for the current contact type
            f_dict[state][contact_type] = []
            f_var_dict[state][contact_type] = []
            n_dict[state][contact_type] = []
            pair_idx_dict[state][contact_type] = []
            shape_dict[state][contact_type] = {}
            
            # Loop through the chromosome / chromosome pairs
            for chrom in h5[state][contact_type]:
                
                # Read the data
                f = h5[state][contact_type][chrom]['f'][:]
                f_var = h5[state][contact_type][chrom]['f_var'][:]
                n = int(h5[state][contact_type][chrom]['nsamples'][...])
                # For nsamples and pair_idx, create a list of the same length as f
                n = np.full(f.size, n).astype(int)
                pair_idx = np.full(f.size, chrom).astype(str)
                
                # Store the flattened arrays in the lists
                f_dict[state][contact_type].append(f.flatten())
                f_var_dict[state][contact_type].append(f_var.flatten())
                n_dict[state][contact_type].append(n)
                pair_idx_dict[state][contact_type].append(pair_idx)
                # Store the shape of the original matrix
                shape_dict[state][contact_type][chrom] = f.shape
                # For intra contacts, store the genomic distances
                if contact_type == 'intra':
                    # Get the genomic distance between all pairs of bins
                    start = index.start[index.chromstr == chrom]
                    gendist = np.abs(start[:, None] - start[None, :])
                    # Extend the list with the flattened genomic distances
                    gendist_dict[state].append(gendist.flatten())

    # Concatenate the lists and convert them to numpy arrays
    for state in states:
        for contact_type in ['intra', 'inter']:
            f_dict[state][contact_type] = np.concatenate(f_dict[state][contact_type])
            f_var_dict[state][contact_type] = np.concatenate(f_var_dict[state][contact_type])
            n_dict[state][contact_type] = np.concatenate(n_dict[state][contact_type])
            pair_idx_dict[state][contact_type] = np.concatenate(pair_idx_dict[state][contact_type])
        gendist_dict[state] = np.concatenate(gendist_dict[state])

    # --- PERFORM THE Z-TEST AGAINST THE AVERAGE IN EACH STATE ---
    
    # Initialize the dictionaries to store the control and the significant contacts
    ctrl_dict = {}
    sig_dict = {}
    
    # Loop over the states / contact types
    for state in states:
        ctrl_dict[state] = {}
        sig_dict[state] = {}
        for contact_type in ['intra', 'inter']:
            
            # Get the contact frequency, variance, and nsamples
            f = f_dict[state][contact_type]
            f_var = f_var_dict[state][contact_type]
            n = n_dict[state][contact_type]
            
            # Create the control array, differently for intra and inter
            # For inter it's simply the mean of f
            if contact_type == 'inter':
                ctrl = np.full(len(f), np.nanmean(f))
            # For intra, we use the mean of f at each genomic distance
            elif contact_type == 'intra':
                ctrl = np.full(len(f), np.nan)
                gendist = gendist_dict[state]
                for d in np.unique(gendist):
                    mask_d = gendist == d
                    # Skip distances with too few data points
                    if np.sum(mask_d) < 100:
                        continue
                    ctrl[mask_d] = np.nanmean(f[mask_d])
            
            # Create a mask to only keep non-NaN values
            valid = np.logical_and(~np.isnan(f), ~np.isnan(f_var))
            valid = np.logical_and(valid, f_var > 0)
            valid = np.logical_and(valid, ~np.isnan(ctrl))
            # Only keep the valid pairs
            f_valid = f[valid]
            f_var_valid = f_var[valid]
            n_valid = n[valid]
            ctrl_valid = ctrl[valid]
            
            # Perform the z-test
            z_valid = (f_valid - ctrl_valid) / np.sqrt(f_var_valid)
            p_valid = stats.norm.sf(z_valid)  # one-tailed p-value
            # Correct p-values using FDR
            _, corr_p_valid, _, _ = multipletests(p_valid, alpha=alpha, method='fdr_tsbh')
            
            # Re-include the NaNs in the p-values
            corr_p = np.full(len(f), np.nan)
            corr_p[valid] = corr_p_valid
            
            # Calculate the effect size of the contacts as the log2 fold change
            eff_size = np.full(len(f), np.nan)
            eff_size[valid] = np.log2(f_valid / ctrl_valid)
            
            # Identify significant contacts
            eff_size_t = eff_size_intra if contact_type == 'intra' else eff_size_inter
            sig = np.logical_and(corr_p < alpha, eff_size > eff_size_t)
            
            # Print the number of significant contacts
            print(f'State: {state}, Contact type: {contact_type}')
            print(f'    Corrected pval < {alpha}: {np.sum(corr_p < alpha)}')
            print(f'    Effect size > {eff_size_t}: {np.sum(eff_size > eff_size_t)}')
            print(f'    Significant loci: {np.sum(sig)}')
            print('\n\n')
            
            # Store the control and significant contacts
            ctrl_dict[state][contact_type] = ctrl
            sig_dict[state][contact_type] = sig


    # --- SAVE THE RESULTS TO THE HDF5 FILE ---

    # Loop over the states / contact types / chromosome pairs
    for state in states:
        for contact_type in ['intra', 'inter']:
            for chrom in h5[state][contact_type]:
                
                # Get the group for the current chromosome pair
                chrom_group = h5[state][contact_type][chrom]
                
                # Get the mask for the current chromosome pair
                mask_chrom = pair_idx_dict[state][contact_type] == chrom
                
                # Get the control and significant contacts for the chromosome pair
                ctrl = ctrl_dict[state][contact_type][mask_chrom]
                sig = sig_dict[state][contact_type][mask_chrom].astype(np.int8)
                # Re-shape to the original matrix shape
                shape = shape_dict[state][contact_type][chrom]
                ctrl = ctrl.reshape(shape)
                sig = sig.reshape(shape)
                
                # If the datasets 'control' or 'significant' already exist, delete them
                if 'control' in chrom_group:
                    del chrom_group['control']
                if 'significant' in chrom_group:
                    del chrom_group['significant']
                
                # Create the datasets for the control and significant contacts
                chrom_group.create_dataset('control', data=ctrl, dtype=np.float64, chunks=True, compression='gzip')
                chrom_group.create_dataset('significant', data=sig, dtype=np.int8, chunks=True, compression='gzip')


# FUNCTION TO IDENTIFY COMPARTMENTS FROM THE CONTACT FREQUENCY MATRICES

def call_compartments(
    h5: h5py.File,
    genden: np.ndarray,
    neigen: int = 5,
    min_var_expl: float = 0.05,
):
    """ Identify compartments from the contact frequency matrices in each state.
    
    For each state / chromosome:
        - compute the observed/expected contact frequency matrix,
        - standardize the observed/expected matrix for each row,
        - compute the correlation matrix,
        - compute the first 'neigen' eigenvectors and eigenvalues of the correlation matrix,
        - select the best eigenvector based on its correlation with gene density, and only if it explains more than 'min_var_expl' of the variance.
        - orient the eigenvector so that it has a positive correlation with gene density,
        - perform L2 normalization on the eigenvector, so that the values are comparable across states and chromosomes.
    
    NOTE: this function requires the control matrices to be present in the H5 file, since it needs to compute the observed/expected contact frequencies.

    Args:
        h5 (h5py.File): H5 file with the contact frequency matrices and the control matrices.
        genden (np.ndarray): gene density array, indexed by the same bins as the Index in the H5 file.
        neigen (int, optional): number of eigenvectors to probe. Defaults to 5.
        min_var_expl (float, optional): minimum fraction of variance explained by the eigenvector to be considered. Defaults to 0.05.
    """
    
    # Read the Index and the states from the H5 file
    index = Index(h5)
    chroms = index.genome.chroms
    states = h5['states'][:].astype(str)
    
    # Make sure that the gene density array is indexed by the same bins as the Index in the H5 file
    if len(genden) != len(index):
        raise ValueError("The gene density array must be indexed by the same bins as the Index in the H5 file.")
    
    # Index the Index chromosome masks and the gene density array by chromosome,
    # so we don't have to do it repeatedly for each chromosome in the loop below
    mask_bychrom = {chrom: index.chromstr == chrom for chrom in chroms}
    genden_bychrom = {chrom: genden[mask_bychrom[chrom]] for chrom in chroms}
    
    
    # --- PERFORM COMPARTMENT CALLING FOR EACH CHROMOSOME IN EACH STATE ---
    
    # Initialize dictionaries of arrays to store:
    #   - the eigenvectors for each state, arrays of shape (nloci,) [same as len(index)]
    #   - the correlation with gene density for each state and chromosome, arrays of shape (nchroms,)
    #   - the variance explained by the eigenvector for each state and chromosome, arrays of shape (nchroms,)
    #   - the O/E matrices for each state and chromosome, arrays of shape (nloci_chrom, nloci_chrom)
    #   - the correlation matrices for each state and chromosome, arrays of shape (nloci_chrom, nloci_chrom)
    results = {}
    for state in states:
        results[state] = {
            'eigenvector': np.full(len(index), np.nan),
            'gd_corr': np.full(len(chroms), np.nan),
            'var_expl': np.full(len(chroms), np.nan),
            'oe_mat': {},
            'corr_mat': {},
        }
    
    # Loop over the states and chromosomes
    for state in states:
        for chromnum, chrom in enumerate(chroms):
            
            # Get the contact frequency and control matrices
            f = h5[state]['intra'][chrom]['f'][:]  # (nloci_chrom, nloci_chrom)
            try:
                ctrl = h5[state]['intra'][chrom]['control'][:]  # (nloci_chrom, nloci_chrom)
            except KeyError:
                # If the control matrix is not available, we cannot compute the observed/expected,
                # so we raise an error
                raise KeyError(f"Control matrix not found. Please run 'call_significant_contacts' first")
            
            # Compute the observed/expected contact frequency matrix
            oe = (f + 1e-10) / (ctrl + 1e-10)  # add a small value to avoid division by zero
            
            # Standardize the observed/expected for each row
            mu = np.nanmean(oe, axis=1, keepdims=True)
            sd = np.nanstd(oe, axis=1, keepdims=True)
            oe_std = (oe - mu) / sd
            
            # Set the diagonal to 0, since np.corrcoef cannot handle NaNs
            np.fill_diagonal(oe_std, 0)
            # Set NaN and Infs to 0, since np.corrcoef cannot handle NaNs
            oe_std[~np.isfinite(oe_std)] = 0
            
            # Compute the correlation matrix
            corr = np.corrcoef(oe_std)
            # Symmetrize the correlation matrix
            corr = (corr + corr.T) / 2
            
            # Store the O/E and correlation matrices in the results dictionary
            results[state]['oe_mat'][chrom] = oe
            results[state]['corr_mat'][chrom] = corr
            
            # Get the first 'neigen' eigenvectors and eigenvalues
            vals, vecs = eigsh(corr, k=neigen, which='LA')  # largest algebraic eigenvalue
            vals = vals[::-1]
            vecs = vecs[:, ::-1]
            
            # Find the best eigenvector based on correlation with gene density
            best = {
                'gd_corr': -1,
                'eigenvector': None,
                'var_expl': None,
            }
            for i in range(neigen):
                v = vecs[:, i].copy()
                
                # Get the fraction of variance explained by the eigenvector
                var_expl = vals[i] / corr.shape[0]
                # Skip if explains less than the minimal required fraction
                if var_expl < min_var_expl:
                    continue
                
                # Orient gene density
                r_gd = clean_correlation(v, genden_bychrom[chrom], return_p=False)
                if r_gd < 0:
                    v = -v
                    r_gd = -r_gd
                
                # Perform L2 normalization
                v = v / np.linalg.norm(v[~np.isnan(v)])
                
                # Set as best if better correlation with gene density
                if r_gd > best['gd_corr']:
                    best['gd_corr'] = r_gd
                    best['eigenvector'] = v
                    best['var_expl'] = var_expl
            
            # If no eigenvector is good enough, we skip this chromosome (leave the results as NaN)
            if best['eigenvector'] is None:
                continue
            # Store the results
            results[state]['eigenvector'][mask_bychrom[chrom]] = best['eigenvector']
            results[state]['gd_corr'][chromnum] = best['gd_corr']
            results[state]['var_expl'][chromnum] = best['var_expl']
    
    
    # --- SAVE THE RESULTS TO THE HDF5 FILE ---
    
    for state in states:
        
        # Get the group for the state
        state_group = h5[state]
        
        # If the group 'compartments' already exists, delete it
        if 'compartments' in state_group:
            del state_group['compartments']
        # Create a group for the compartments
        comp_group = state_group.create_group('compartments')
        
        # Save the eigenvector, gene density correlation, and variance explained
        comp_group.create_dataset('eigenvector', data=results[state]['eigenvector'], dtype=np.float64, chunks=True, compression='gzip')
        comp_group.create_dataset('gd_corr', data=results[state]['gd_corr'], dtype=np.float64)
        comp_group.create_dataset('var_expl', data=results[state]['var_expl'], dtype=np.float64)
        
        # Create a group for each chromosome, and save the O/E and correlation matrices
        for chrom in chroms:
            # Skip if the OE or corr matrices are not available for this chromosome
            if chrom not in results[state]['oe_mat'] or chrom not in results[state]['corr_mat']:
                continue
            # Otherwise, create a group for the chromosome and save the matrices
            chrom_group = comp_group.create_group(chrom)
            chrom_group.create_dataset('oe_mat', data=results[state]['oe_mat'][chrom], dtype=np.float64, chunks=True, compression='gzip')
            chrom_group.create_dataset('corr_mat', data=results[state]['corr_mat'][chrom], dtype=np.float64, chunks=True, compression='gzip')
