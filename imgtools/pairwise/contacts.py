import os
import h5py
import numpy as np
from scipy.spatial import cKDTree
from alabtools.utils import map_indices
from ..cte import ChromatinTracingExperiment
from .pairwise_utils import *
from . import parallel_pairwise


def calculate_intra_matrices(
    xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, bins: np.ndarray, N: int, thresh: float
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

    Returns:
        cop_mat (np.ndarray): co-presence matrix
        ctc_mat (np.ndarray): contact matrix
    """
    
    # Create the co-presence matrix
    # First we find the bins that are present and those that are not
    present = np.zeros(N, dtype=bool)
    present[np.unique(bins)] = True
    # Then we create the co-presence matrix as the outer product of the present vector
    cop_mat = np.outer(present, present)
    
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
    # Make it binary
    ctc_mat[ctc_mat > 0] = 1
    
    # Remove the diagonal
    np.fill_diagonal(cop_mat, 0)
    np.fill_diagonal(ctc_mat, 0)
    
    return cop_mat, ctc_mat

def calculate_inter_matrices(
    xs_1: np.ndarray, ys_1: np.ndarray, zs_1: np.ndarray, bins_1: np.ndarray,
    xs_2: np.ndarray, ys_2: np.ndarray, zs_2: np.ndarray, bins_2: np.ndarray,
    N_1: int, N_2: int, thresh: float
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

    Returns:
        cop_mat (np.ndarray): co-presence matrix
        cnt_mat (np.ndarray): contact matrix
    """
    
    # Create the co-presence matrix
    # First we find the bins that are present and those that are not
    present_1 = np.zeros(N_1, dtype=bool)
    present_2 = np.zeros(N_2, dtype=bool)
    present_1[np.unique(bins_1)] = True
    present_2[np.unique(bins_2)] = True
    # Then we create the co-presence matrix as the outer product of the present vector
    cop_mat = np.outer(present_1, present_2)
    
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
        
        # Make it binary
        cnt_mat[cnt_mat > 0] = 1
    
    return cop_mat, cnt_mat

def initialize_h5(filename: str, n_1: int, n_2: int, cte: ChromatinTracingExperiment) -> h5py.File:
    """ Initialize the HDF5 file for storing the average matrices.
    
    For each (unique) state label in the CTE, a group is created with the following datasets:
        - 'nsamples': a matrix of shape (n_1, n_2) to count the number of samples in each bin pair
        - 'mean': a matrix of shape (n_1, n_2) to store the mean contact values of each bin pair
        - 'var': a matrix of shape (n_1, n_2) to store the variance of contact values of each bin pair
    
    The 'all' state is also added to the list of states, which will contain the average matrices
    for all cells regardless of their state.

    Args:
        filename (str): name of the HDF5 file to create.
        n_1 (int): number of bins in the first chromosome
        n_2 (int): number of bins in the second chromosome
        cte (ChromatinTracingExperiment)

    Returns:
        (h5py.File): the initialized HDF5 file with groups and datasets for each state.
    """
    
    # Open the HDF5 file for writing
    h5 = h5py.File(filename, 'w')
    
    # Get the unique states from the CTE
    states = np.unique(cte.cell_states)
    # Add the 'all' state to the list of states
    states = np.append(states, 'all')
    
    # Create a group for each state
    for state in states:
        state_group = h5.create_group(state)
            
        # Initialize the matrices for the state group
        state_group.create_dataset('nsamples', (n_1, n_2), dtype=np.int64, chunks=True, compression='gzip')
        state_group.create_dataset('mean', (n_1, n_2), dtype=np.float64, chunks=True, compression='gzip')
        state_group.create_dataset('var', (n_1, n_2), dtype=np.float64, chunks=True, compression='gzip')
    
    return h5

def update_average_matrices(group: h5py.Group, cop_mat: np.ndarray, ctc_mat: np.ndarray) -> None:
    """ Update the average matrices in the group with the new co-presence and contact matrices.
    
    Uses the Welford's method to update the mean and std matrices to avoid numerical instability.
    
    Args:
        group (h5py.Group): group to update
        cop_mat (np.ndarray): co-presence matrix
        ctc_mat (np.ndarray): contact matrix
    """
    
    # If the co-presence matrix is empty, exit
    if not cop_mat.any():
        return
    
    # Get the current matrices
    n = group['nsamples'][...].astype(np.int64, copy=False)
    mean = group['mean'][...].astype(np.float64, copy=False)
    var = group['var'][...].astype(np.float64, copy=False)
    
    # Update the number of samples
    n[cop_mat] += 1
    n_new = n  # alias for clarity
    
    # Update the mean where the co-presence matrix is 1
    delta = ctc_mat - mean  # uses old mean
    mean[cop_mat] += delta[cop_mat] / n_new[cop_mat]
    
    # Update the variance using Welford's method
    # We can only update where the number of samples is 2 or more
    # (and, of course, where the co-presence matrix is 1)
    mask_var = np.logical_and(cop_mat, n_new >= 2)
    if mask_var.any():
        delta_2 = ctc_mat - mean  # uses new mean
        var[mask_var] = (1 / (n_new[mask_var] - 1)) * ((n_new[mask_var] - 2) * var[mask_var] + delta[mask_var] * delta_2[mask_var])
    
    # Update the group with the new matrices
    group['nsamples'][...] = n
    group['mean'][...] = mean
    group['var'][...] = var


def func_node(chrom_1: str, chrom_2: str, cte_name: str, config: dict, tempdir: str) -> None:
    
    # Read the CTE file and its index
    cte = ChromatinTracingExperiment(cte_name, 'r')
    cte_index = cte.index
    
    # Read the target index from the config
    index = read_target_index(cte, config)
    # Get the target index hashmap
    index_hashmap = index.get_index_hashmap()
    # Get the offsets and the lengths of the chromosomes in the target Index
    ind_chrom_1 = np.where(index.genome.chroms == chrom_1)[0][0]
    ind_chrom_2 = np.where(index.genome.chroms == chrom_2)[0][0]
    offset_chrom_1, offset_chrom_2 = index.offset[ind_chrom_1], index.offset[ind_chrom_2]
    size_chrom_1, size_chrom_2 = index.chrom_sizes[ind_chrom_1], index.chrom_sizes[ind_chrom_2]
    
    # Map the CTE Index to the target Index:
    #   index_map = {(chrom, cte_start, cte_end): [(chrom, start, end)], ...}
    cte_to_index_map = map_indices(cte_index, index)
    
    # Initialize the HDF5 file for storing the average matrices
    filename = os.path.join(tempdir, f'{chrom_1}_{chrom_2}.h5')
    h5 = initialize_h5(filename, size_chrom_1, size_chrom_2, cte)
    
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
            xs_1, ys_1, zs_1, cte_starts_1, cte_ends_1, _, _ = cte.get_data(cellID, chrom_1, traceID_1, format='numpy')
            
            # Convert the domain info (chrom, start, end) of each spot
            # into its bin position along the target Index
            bins_1 = get_bins(chrom_1, cte_starts_1, cte_ends_1, cte_to_index_map, index_hashmap)
            # Remove the offset of the chromosome, so that bins_1 start from 0
            bins_1 = bins_1 - offset_chrom_1
            
            # If chrom_1 = chrom_2, this is an intra-chromosomal contact calculation
            # and we don't need to loop over the traceIDs of chrom_2.
            if chrom_1 == chrom_2:
                
                # Calculate the co-presence and contact matrices for the intra-chromosomal case
                cop_mat, cnt_mat = calculate_intra_matrices(
                    xs_1, ys_1, zs_1, bins_1,
                    size_chrom_1, config['thresh']
                )
                
                # Update the average matrices in the HDF5 file
                update_average_matrices(h5[state], cop_mat, cnt_mat)
                update_average_matrices(h5['all'], cop_mat, cnt_mat)
                
                continue
            
            # Otherwise, with chrom_1 != chrom_2, this is an inter-chromosomal contact calculation
            # Loop over the traceIDs of chromosome 2 in the cell
            for traceID_2 in traceID_map[chrom_2]:
                
                # Get the data of chrom_2 / traceID_2 in the cell
                xs_2, ys_2, zs_2, cte_starts_2, cte_ends_2, _, _ = cte.get_data(cellID, chrom_2, traceID_2, format='numpy')
                
                # Get the bins of chrom_2 as before
                bins_2 = get_bins(chrom_2, cte_starts_2, cte_ends_2, cte_to_index_map, index_hashmap)
                bins_2 = bins_2 - offset_chrom_2  # Remove the offset of the chromosome
                
                # Calculate the co-presence and contact matrices
                cop_mat, cnt_mat = calculate_inter_matrices(
                    xs_1, ys_1, zs_1, bins_1,
                    xs_2, ys_2, zs_2, bins_2,
                    size_chrom_1, size_chrom_2,
                    config['thresh']
                )
                
                # Update the average matrices in the HDF5 file
                update_average_matrices(h5[state], cop_mat, cnt_mat)
                update_average_matrices(h5['all'], cop_mat, cnt_mat)

def reduce_initialization(_, cte_name: str, config: dict) -> None:
    
    # Open the CTE file
    cte = ChromatinTracingExperiment(cte_name, 'r')
    
    # Get the unique states from the CTE
    states = np.unique(cte.cell_states)
    # Add the 'all' state to the list of states
    states = np.append(states, 'all')
    
    # Open the HDF5 file for writing
    h5 = h5py.File(config['filename'], 'w')
    
    # Create a group for each state
    for state in states:
        state_group = h5.create_group(state)
        
        # Create an intra and an inter group
        intra_group = state_group.create_group('intra')
        inter_group = state_group.create_group('inter')
    
    h5.close()

def reduce_update(chrom_1: str, chrom_2: str, result: dict, pair_result: dict, cte_name: str, config: dict, tempdir: str) -> None:
    
    # Open the CTE file
    cte = ChromatinTracingExperiment(cte_name, 'r')
    
    # Get the unique states from the CTE
    states = np.unique(cte.cell_states)
    # Add the 'all' state to the list of states
    states = np.append(states, 'all')
    
    # Read the h5 file for the collected matrices
    try:
        h5 = h5py.File(config['filename'], 'a')
    except OSError as e:
        raise OSError(f"Error opening HDF5 file {config['filename']}: {e}")
    
    # Read the h5 file for the chromosome pair matrices
    try:
        h5_pair = h5py.File(os.path.join(tempdir, f'{chrom_1}_{chrom_2}.h5'), 'r')
    except OSError as e:
        raise OSError(f"Error opening HDF5 file for chromosome pair {chrom_1}_{chrom_2}: {e}")

    # Loop over the states
    for state in states:
        
        # If the chromosome pair is intra-chromosomal, we use the 'intra' group
        if chrom_1 == chrom_2:
            group = h5[state]['intra']
        # Otherwise, we use the 'inter' group
        else:
            group = h5[state]['inter']
        
        # Create a group for the chromosome pair
        pair_group = group.create_group(f'{chrom_1}_{chrom_2}')
        
        # Copy the matrices from the pair result to the pair group
        pair_group.create_dataset('nsamples', data=h5[state]['nsamples'][...], dtype=np.int64, chunks=True, compression='gzip')
        pair_group.create_dataset('mean', data=h5[state]['mean'][...], dtype=np.float64, chunks=True, compression='gzip')
        pair_group.create_dataset('var', data=h5[state]['var'][...], dtype=np.float64, chunks=True, compression='gzip')
    
    h5.close()
    h5_pair.close()

def main(cte: ChromatinTracingExperiment, config: dict) -> None:
    parallel_pairwise.control_func(cte, config, func_node, reduce_initialization, reduce_update)
