# Functions for computing metrics (e.g. neighbor distances) for a ChromatinTracingExperiment object

import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
from .cte import ChromatinTracingExperiment
from . import cte_utils
from . import cte_parallel


def get_trace_ranks_for_chromosome(cte: ChromatinTracingExperiment, cellID: str, chrom: str) -> dict:
    """ Get the ranks (by n. of spots) of the traces in a chromosome.
    
        The rank of valid traces is positive:
            rank 1 --> valid trace with the most spots, 2 --> second most, etc.
        The rank of valid traces is negative:
            rank -1 --> noisy trace with the most spots, -2 --> second most, etc.

    Args:
        cte (ChromatinTracingExperiment)
        cellID (str)
        chrom (str)

    Returns:
        ranks (dict): dictionary of the rank of traces for a chromosome in a cell.
                        ranks[traceID] = rank
    """
    
    # Initialize the counts of spots per trace for valid and noisy traces
    valid_counts, noisy_counts = {}, {}
    
    # Take the chromosome data
    chrom_data = cte.get_data(cellID, chrom, format='dict')
    
    # Loop over traces and count the number of spots
    for traceID in chrom_data:
        
        if cte.look_for_noisy_trace(traceID):
            noisy_counts[traceID] = len(chrom_data[traceID])
        else:
            valid_counts[traceID] = len(chrom_data[traceID])

    # Sort traces by number of spots
    sorted_valid = sorted(valid_counts, key=valid_counts.get, reverse=True)
    sorted_noisy = sorted(noisy_counts, key=noisy_counts.get, reverse=True)
    
    # Assign rank to traces
    ranks = {}
    # Rank valid traces (1, 2, 3, ...)
    for rank, traceID in enumerate(sorted_valid, start=1):
        ranks[traceID] = rank
    # Rank noisy traces (-1, -2, -3, ...)
    for rank, traceID in enumerate(sorted_noisy, start=1):
        ranks[traceID] = -rank

    return ranks


def get_trace_ranks_for_cell(cte: ChromatinTracingExperiment, cellID: str) -> dict:
    """ Get ranks of traces - within each chromosome - for a cell.
    
    Within each chromosome, the rank of valid traces is positive:
        rank 1 --> valid trace with the most spots, 2 --> second most, etc.
    Within each chromosome, the rank of valid traces is negative:
        rank -1 --> noisy trace with the most spots, -2 --> second most, etc.

    Args:
        cte (ChromatinTracingExperiment)
        cellID (str)

    Returns:
        ranks (dict): dictionary of the rank of traces for a cell.
                        ranks[chrom][traceID] = rank
    """
    
    ranks = {}
    
    cell_data = cte.get_data(cellID, format='dict')
    
    for chrom in cell_data:
        ranks[chrom] = get_trace_ranks_for_chromosome(cte, cellID, chrom)
    
    return ranks


def distribution_nspot_per_trace(cte: ChromatinTracingExperiment, ignore_noisy_trace: bool = True) -> np.ndarray:
    """ Computes the distribution of the number of spots per trace across cells.

    Args:
        cte (ChromatinTracingExperiment)
        ignore_noisy_trace (bool, optional): ignore noisy traces. Defaults to True.

    Returns:
        nspot_per_trace (np.array): array of the number of spots per trace across cells.
    """
    
    nspot_per_trace = []
    
    for cellID in cte.cell_labels:
        cell_data = cte.get_data(cellID, format='dict')
        
        for chrom in cell_data:
            for traceID in cell_data[chrom]:
                
                if ignore_noisy_trace and cte.look_for_noisy_trace(traceID):
                    continue
                
                nspot = len(cell_data[chrom][traceID])
                nspot_per_trace.append(nspot)
    
    nspot_per_trace = np.array(nspot_per_trace)
    
    return nspot_per_trace


def distirbution_avg_spot_per_tracerank(cte: ChromatinTracingExperiment) -> dict:
    """ Computes the average number of spots per trace rank.
    
    Within each chromosome, the rank of valid traces is positive:
        rank 1 --> valid trace with the most spots, 2 --> second most, etc.
    Within each chromosome, the rank of valid traces is negative:
        rank -1 --> noisy trace with the most spots, -2 --> second most, etc.
    
    Args:
        cte (ChromatinTracingExperiment)

    Returns:
        nspot_per_trank (dict): dictionary of the average number of spots per trace rank."""
    
    # Initialize dictionary for the average number of spots per trace rank
    # We initialize the default element to an empty list, so that we can append to it without checking if it exists
    nspot_per_rank = defaultdict(list)

    for cellID in cte.cell_labels:
        cell_data = cte.get_data(cellID, format='dict')
        
        for chrom in cell_data:
            
            # Get ranks of traces in the chromosome
            trace_ranks = get_trace_ranks_for_chromosome(cte, cellID, chrom)
            
            for traceID in cell_data[chrom]:
                
                # rank of traceID
                r = trace_ranks[traceID]
                # Number of spots in traceID
                nspot = len(cell_data[chrom][traceID])
                # Add nspot_t to the list of spots for rank t
                nspot_per_rank[r].append(nspot)
    
    # Compute average number of spots per trace rank
    for r in nspot_per_rank:
        nspot_per_rank[r] = np.mean(np.array(nspot_per_rank[r]))

    return nspot_per_rank


def distribution_ntrace_per_chromosome(cte: ChromatinTracingExperiment, ignore_noisy_trace: bool = True) -> np.ndarray:
    """Computes the distribution of the number of traces per chromosome across cells.
    
    Args:
        cte (ChromatinTracingExperiment)
        ignore_noisy_trace (bool, optional): ignore noisy traces. Defaults to True.

    Returns:
        ntrace_per_chrom (np.ndarray): list of the number of traces per chromosome across cells."""
    
    ntrace_per_chrom = []  # list of the number of traces per chromosome across cells

    for cellID in cte.cell_labels:
        cell_data = cte.get_data(cellID, format='dict')
        
        for chrom in cell_data:
            
            ntrace_chrom_cell = 0
            
            for traceID in cell_data[chrom]:
                
                if ignore_noisy_trace and cte.look_for_noisy_trace(traceID):
                    continue
                
                ntrace_chrom_cell += 1
                
            ntrace_per_chrom.append(ntrace_chrom_cell)
            
    ntrace_per_chrom = np.array(ntrace_per_chrom)
    
    return ntrace_per_chrom


def compute_trace_coverage(cte: ChromatinTracingExperiment, cellID: str, chrom: str, traceID: str) -> float:
    """ Computes the coverage of a trace.
    
    The coverage is defined as the number of unique domains divided by the total number of domains.

    Args:
        cte (ChromatinTracingExperiment)
        cellID (str)
        chrom (str)
        traceID (str)

    Returns:
        coverage (float): coverage of the trace.
    """
    
    # Get the data of the trace
    trace_data = cte.get_data(cellID, chrom, traceID, format='dict')
    
    # Find unique domains in traceID
    unique_domains = set()
    for spotID in trace_data:
        spot_data = trace_data[spotID]
        start = spot_data['start']
        end = spot_data['end']
        unique_domains.add((start, end))
    
    # The coverage is the number of unique domains divided by the number of domains
    coverage = len(unique_domains) / np.sum(cte.index.chromstr == chrom)
    
    return coverage


def distribution_coverage_per_trace(cte: ChromatinTracingExperiment, ignore_noisy_traces: bool = True) -> np.ndarray:
    """ Compute the distribution of the coverage of each trace.
    
    Args:
        cte (ChromatinTracingExperiment)
        ignore_noisy_traces (bool, optional): ignore noisy traces. Defaults to True.

    Returns:
        coverage_distribution (np.array): array of the coverage of each trace.
    """

    coverage_distribution = []
    
    for cellID in cte.cell_labels:
        cell_data = cte.get_data(cellID, format='dict')
        
        for chrom in cell_data:
            for traceID in cell_data[chrom]:
                
                # ignore noisy traces if requested
                if ignore_noisy_traces and cte.look_for_noisy_trace(traceID):
                    continue
                
                coverage = compute_trace_coverage(cte, cellID, chrom, traceID)
                
                # add coverage to list
                coverage_distribution.append(coverage)
    
    coverage_distribution = np.array(coverage_distribution)
    
    return coverage_distribution


def compute_trace_neighbor_distances(cte: ChromatinTracingExperiment, cellID: str, chrom: str, traceID: str) -> (np.ndarray, np.ndarray):
    """ Computes the genomic and spatial distances between neighboring spots in a trace.

    Args:
        cte (ChromatinTracingExperiment)
        cellID (str)
        chrom (str)
        traceID (str)

    Returns:
        gdist (np.array): array of the genomic distances between neighboring spots in the trace.
        sdist (np.array): array of the spatial distances between neighboring spots in the trace.
    """
    
    # Check that cellID, chrom and traceID are in the data
    if cellID not in cte.data:
        raise ValueError("cellID {} not in data.".format(cellID))
    if chrom not in cte.data[cellID]:
        raise ValueError("chrom {} not in data[{}].".format(chrom, cellID))
    if traceID not in cte.data[cellID][chrom]:
        raise ValueError("traceID {} not in data[{}][{}].".format(traceID, cellID, chrom))
    
    # get the data in numpy array format
    xs, ys, zs, chroms, starts, ends, lums, spotIDs = cte.get_data(cellID, chrom, traceID, format='numpy')
    crds = np.array([xs, ys, zs]).T
    
    # If there is only one spot, skip
    if len(crds) == 1:
        return None, None
    
    # Sort by genomic start position
    crds = crds[np.argsort(starts)]
    starts = starts[np.argsort(starts)]
    
    # Compute genomic distances between neighboring spots
    gdist = np.diff(starts)
    
    # Compute spatial distances between neighboring spots
    sdist = np.linalg.norm(np.diff(crds, axis=0), axis=1)
    
    return gdist, sdist


def distribution_neighbor_distances(cte: ChromatinTracingExperiment, ignore_noisy_traces: bool = True) -> dict:
    """ Compute the average spatial and genomic distance between neighboring spots in each trace.

    Args:
        cte (ChromatinTracingExperiment)
        ignore_noisy_traces (bool, optional): ignore noisy traces. Defaults to True.

    Returns:
        distance_distributions (dict): dictionary of the average spatial and genomic distance between neighboring spots in each trace.
                                    Contains the following keys:
                                        - avg_genomic_distances
                                        - max_genomic_distances
                                        - min_genomic_distances
                                        - avg_spatial_distances
                                        - max_spatial_distances
                                        - min_spatial_distances
    """
    
    # Initialize lists
    avg_genomic_distances = []
    max_genomic_distances = []
    min_genomic_distances = []
    
    avg_spatial_distances = []
    max_spatial_distances = []
    min_spatial_distances = []
    
    # Loop over cells, chromosomes and traces and fill lists
    for cellID in cte.cell_labels:
        cell_data = cte.get_data(cellID, format='dict')
        
        for chrom in cell_data:
            for traceID in cell_data[chrom]:
                
                # ignore noisy traces if requested
                if ignore_noisy_traces and cte.look_for_noisy_trace(traceID):
                    continue
                
                # get the genomic and spatial distances between neighboring spots in the trace
                gdist, sdist = compute_trace_neighbor_distances(cte, cellID, chrom, traceID)
                
                # Add to lists
                avg_genomic_distances.append(np.mean(gdist))
                max_genomic_distances.append(np.max(gdist))
                min_genomic_distances.append(np.min(gdist))
                
                avg_spatial_distances.append(np.mean(sdist))
                max_spatial_distances.append(np.max(sdist))
                min_spatial_distances.append(np.min(sdist))
    
    # Return lists (cast to numpy arrays) in dictionary
    distance_distributions = {
        'avg_genomic_distances': np.array(avg_genomic_distances),
        'max_genomic_distances': np.array(max_genomic_distances),
        'min_genomic_distances': np.array(min_genomic_distances),
        'avg_spatial_distances': np.array(avg_spatial_distances),
        'max_spatial_distances': np.array(max_spatial_distances),
        'min_spatial_distances': np.array(min_spatial_distances)
    }
    
    return distance_distributions


# Distribution of distances between sister chromatids

def run_sisterdist_parallel(cte: ChromatinTracingExperiment, config: dict) -> dict:
    """ Run the sister distance task in parallel.

    Args:
        cte (ChromatinTracingExperiment)

    Returns:
        (dict): dictionary with the distances between sister chromatids for each cell, plus aggregated distances.
                Particularly, the dictionary has a key for each cell, whose values are arrays of sister distances, plus the following:
                - 'all': array of the distances between sister chromatids for all cells
                - 'all_G1' (optional): array of the distances between sister chromatids for cells in G1
                - 'all_S' (optional): array of the distances between sister chromatids for cells in S
                - 'all_G2' (optional): array of the distances between sister chromatids for cells in G2
                ('all_G1', 'all_S' and 'all_G2' are only present if the CTE has a 'cell_states' array with 'G1', 'S' and 'G2')
    """
    
    sisterdist = cte_parallel.control_func(
        cte,
        config,
        {},  # no required keys needed
        sisterdist_nfunc,
        sisterdist_rfunc_init,
        sisterdist_rfunc_update
    )
    
    return sisterdist

def sisterdist_nfunc(cellID: str, cte_name: str, _) -> np.ndarray:
    """ Node-level function to calculate the distances between sister chromatids for a cell.

    Args:
        cellID (str)
        cte_name (str): name of the ChromatinTracingExperiment
        _: not used, just to match the signature of the function

    Returns:
        np.ndarray: array of the distances between sister chromatids for the current cell.
    """
    
    cte = ChromatinTracingExperiment(cte_name, 'r')
    
    # Get the data of the cell
    cell_data = cte.get_data(cellID, format='dict')
    
    # Initialize the array of distances between sister chromatids
    cell_sisterdist = np.array([])
    
    # Loop over chromosomes and traces
    for chrom in cell_data:
        for traceID in cell_data[chrom]:
            
            # Get the data of the trace
            trace_data = cell_data[chrom][traceID]
            
            # Convert the data to numpy arrays
            xs, ys, zs, starts, _, _, _ = cte_utils.trace_dict_to_numpy(trace_data)
            
            # Get the 3D distances between each pair of spots
            crds = np.array([xs, ys, zs]).T
            dists = cdist(crds, crds)
            
            # Get a matrix that is True if the spots have the same start position
            same_start = np.equal.outer(starts, starts)
            # Avoid double counting, setting the lower triangle to False
            same_start = np.triu(same_start)
            # Set the diagonal to False
            np.fill_diagonal(same_start, False)
            
            # Get the distances between sister chromatids
            sisterdist = dists[same_start]
            
            # Add the distances to the array
            cell_sisterdist = np.concatenate((cell_sisterdist, sisterdist))
            
            del trace_data, xs, ys, zs, starts, crds, dists, same_start, sisterdist
    
    return cell_sisterdist

def sisterdist_rfunc_init(_1, _2, _3) -> dict:
    """ Initialize the sister distance dictionary for the reduce function.
    
    It's an empty dictionary, with a key 'all' that contains an empty array.

    Args:
        _*: not used, just to match the signature of the function

    Returns:
        (dict): dictionary of the distances between sister chromatids for each cell.
    """
    sisterdist = {
        'all': np.array([]),
    }
    return sisterdist

def sisterdist_rfunc_update(cellID: str, sisterdist: dict, cell_sisterdist: np.ndarray, cte_name: str, _2) -> dict:
    """ Update the sister distance dictionary for the reduce function.
    
    Adds the distances of cellID to the dictionary, and updates the aggregated distances by concatenating the cell distances.
    
    If the CTE has a cell_states array with 'G1', 'S' and 'G2', it also appends the cell distances to the appropriate state.

    Args:
        cellID (str)
        sisterdist (dict): dictionary of the distances between sister chromatids for each cell.
        cell_sisterdist (np.ndarray): array of the distances between sister chromatids for the current cell.
        cte_name (str): name of the ChromatinTracingExperiment
        _*: not used, just to match the signature of the function

    Returns:
        (dict): updated sister distance dictionary.
    """
    # Add the distances of cellID to the dictionary
    sisterdist[cellID] = cell_sisterdist
    
    # Update the aggregated distances
    sisterdist['all'] = np.concatenate((sisterdist['all'], cell_sisterdist))
    
    # Load the ChromatinTracingExperiment
    cte = ChromatinTracingExperiment(cte_name, 'r')
    
    # If the CTE doesn't have a 'cell_states' data,
    # or if the cell_states is not uniquely made of 'G1', 'S' and 'G2',
    # exit the function
    if 'cell_states' not in cte:
        return sisterdist
    if set(cte.cell_states) != set(['G1', 'S', 'G2']):
        return sisterdist
    
    # Otherwise, append the cell_sisterdist to the appropriate state
    
    # Get the state of the cell
    cellnum = cte.get_cellnum(cellID)
    cellstate = cte.cell_states[cellnum]
    
    # Append the cell_sisterdist to the appropriate state (create the key if it doesn't exist)
    state_key = 'all_{}'.format(cellstate)
    if state_key not in sisterdist:
        sisterdist[state_key] = np.array([])
    sisterdist[state_key] = np.concatenate((sisterdist[state_key], cell_sisterdist))
    
    return sisterdist


# Homologues proximity

def run_homoprox_parallel(cte: ChromatinTracingExperiment, config: dict) -> dict:
    """ Run the homologues proximity task in parallel.

    Args:
        cte (ChromatinTracingExperiment)
        config (dict)
    
    Returns:
        homoprox_ratio (dict): homologous proximity ratio for each chromosome.
    """
    
    def rfunc_init(_1, _2, _3, _4, _5) -> dict:
        """ Initialize the homologous proximity ratio dictionary for the reduce function.

        Args:
            _*: not used, just to match the signature of the function

        Returns:
            homoprox (dict): dictionary of the homologous proximity count for each chromosome.
                                homoprox['prox_count'][chrom] = number of chromosomes with proximal homologues
                                homoprox['total_count'][chrom] = total number of chromosomes
                                homoprox['ratio'][chrom] = homologous proximity ratio for the chromosome
        """
        
        homoprox = {
            'prox_count': {},
            'total_count': {},
            'ratio': {}
        }
        
        return homoprox
    
    def rfunc_update(_1, homoprox: dict, cell_homoprox: dict, _2, _3, _4, _5, _6) -> dict:
        """ Update the homologous proximity ratio dictionary for the reduce function.

        Args:
            homoprox (dict): dictionary of the homologous proximity count for each chromosome.
            cell_homoprox (dict): for each chromosome in the cell, False if no homologues are close, True if they are
            _*: not used, just to match the signature of the function

        Returns:
            (dict): updated homologous proximity ratio dictionary.
        """
        
        for chrom in cell_homoprox:
            
            # Increment the total count for the current chromosome
            if chrom not in homoprox['total_count']:
                homoprox['total_count'][chrom] = 0
            homoprox['total_count'][chrom] = 1
            
            # If the homologues are proximal, increment the proximal count for the current chromosome
            if cell_homoprox[chrom]:
                if chrom not in homoprox['prox_count']:
                    homoprox['prox_count'][chrom] = 0
                homoprox['prox_count'][chrom] += 1
            
            # Update the ratio for the current chromosome
            homoprox['ratio'][chrom] = homoprox['prox_count'][chrom] / homoprox['total_count'][chrom]

        return homoprox
    
    homoprox = cte_parallel.control_func(
        cte,
        config,
        homoprox_required_keys,
        homoprox_nfunc,
        rfunc_init,
        rfunc_update
    )
    
    return homoprox['ratio']

homoprox_required_keys = {
    'proximity_threshold': {'type': float, 'positive': True},
    'use': {
        'data': True,
        'index': False,
        'alphashapes': False
    }
}

def homoprox_nfunc(_1, cell_data: dict, _2, _3, _4, config: dict) -> dict:
    """ Cell-level function for the homologous proximity task, to be executed on a node.
    It checks if the homologues of each chromosome are close, and returns a dictionary with the result.

    Args:
        cell_data (dict)
        config (dict)
        _*: not used, just to match the signature of the function

    Returns:
        cell_homoprox (dict): for each chromosome in the cell, False if no homologues are close, True if they are
    """
    
    # Initialize output
    cell_homoprox = {}  # for each chromosome, False if no homologues are close, True if they are
    
    # Loop over chromosomes
    for chrom in cell_data:

        # Skip chromosomes with less than 2 traces
        if len(cell_data[chrom]) < 2:
            continue
        
        # Take the data of the current chromosome
        chrom_data = cell_data[chrom]
        
        # Initialize the proximum boolean for the current chromosome
        are_proximal = False  # True if the homologues of the chromosome are close, False otherwise
        
        # Loop over traces in the chromosome and check if they are close
        for i1, traceID_1 in enumerate(chrom_data):
            
            # If we have already found a pair of proximal homologues, we can stop
            if are_proximal:
                break
            
            for i2, traceID_2 in enumerate(chrom_data):
                
                # Avoid comparing the same pair of traces twice (and avoid comparing a trace to itself)
                if i1 >= i2:
                    continue
                
                # Convert the data to numpy arrays
                xs1, ys1, zs1, _, _, _, _, _ = cte_utils.trace_dict_to_numpy(chrom_data[traceID_1])
                xs2, ys2, zs2, _, _, _, _, _ = cte_utils.trace_dict_to_numpy(chrom_data[traceID_2])
                
                # Calculate the minimum distance between the two traces
                crd1 = np.array([xs1, ys1, zs1]).T
                crd2 = np.array([xs2, ys2, zs2]).T
                min_dist = np.min(cdist(crd1, crd2))
                
                # If the minimum distance is below the threshold, we have found a pair of proximal homologues
                # We can stop looping over traces
                if min_dist <= config['proximity_threshold']:
                    are_proximal = True
                    break
        
        # Save the result of the current chromosome
        cell_homoprox[chrom] = are_proximal
    
    return cell_homoprox
