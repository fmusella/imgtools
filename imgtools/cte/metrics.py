# Functions for computing metrics (e.g. neighbor distances) for a ChromatinTracingExperiment object

import numpy as np
from collections import defaultdict
from .cte import ChromatinTracingExperiment
from . import utils


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
    
    for traceID in cte.data[cellID][chrom]:
        
        if cte.look_for_noisy_trace(traceID):
            noisy_counts[traceID] = len(cte.data[cellID][chrom][traceID])
        else:
            valid_counts[traceID] = len(cte.data[cellID][chrom][traceID])

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
    
    for chrom in cte.data[cellID]:
        ranks[chrom] = get_trace_ranks_for_chromosome(cellID, chrom)
    
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
    
    for cellID in cte.data:
        for chrom in cte.data[cellID]:
            for traceID in cte.data[cellID][chrom]:
                
                if ignore_noisy_trace and cte.look_for_noisy_trace(traceID):
                    continue
                
                nspot = len(cte.data[cellID][chrom][traceID])
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

    for cellID in cte.data:
        for chrom in cte.data[cellID]:
            
            # Get ranks of traces in the chromosome
            trace_ranks = get_trace_ranks_for_chromosome(cellID, chrom)
            
            for traceID in cte.data[cellID][chrom]:
                
                # rank of traceID
                r = trace_ranks[traceID]
                # Number of spots in traceID
                nspot = len(cte.data[cellID][chrom][traceID])
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

    for cellID in cte.data:
        for chrom in cte.data[cellID]:
            
            ntrace_chrom_cell = 0
            
            for traceID in cte.data[cellID][chrom]:
                
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
    
    # Check that cellID, chrom and traceID are in the data
    if cellID not in cte.data:
        raise ValueError("cellID {} not in data.".format(cellID))
    if chrom not in cte.data[cellID]:
        raise ValueError("chrom {} not in data[{}].".format(chrom, cellID))
    if traceID not in cte.data[cellID][chrom]:
        raise ValueError("traceID {} not in data[{}][{}].".format(traceID, cellID, chrom))
    
    # Find unique domains in traceID
    unique_domains = set()
    for spotID in cte.data[cellID][chrom][traceID]:
        start = cte.data[cellID][chrom][traceID][spotID]['start']
        end = cte.data[cellID][chrom][traceID][spotID]['end']
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
    
    for cellID in cte.data: 
        for chrom in cte.data[cellID]:
            for traceID in cte.data[cellID][chrom]:
                
                # ignore noisy traces if requested
                if ignore_noisy_traces and cte.look_for_noisy_trace(traceID):
                    continue
                
                coverage = compute_trace_coverage(cellID, chrom, traceID)
                
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
    xs, ys, zs, chroms, starts, ends, lums, spotIDs = utils.trace_dict_to_numpy(cte.data[cellID][chrom][traceID])
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
    for cellID in cte.data:
        for chrom in cte.data[cellID]:
            for traceID in cte.data[cellID][chrom]:
                
                # ignore noisy traces if requested
                if ignore_noisy_traces and cte.look_for_noisy_trace(traceID):
                    continue
                
                # get the genomic and spatial distances between neighboring spots in the trace
                gdist, sdist = compute_trace_neighbor_distances(cellID, chrom, traceID)
                
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


def run_homologues_proximity(cte, config: dict):
    # FIX ONCE PARALLELIZATION IS SORTED OUT
    
    # Create a temporary directory
    tempdir = tempfile.mkdtemp(dir=os.getcwd())
    sys.stdout.write("Temporary directory for nodes' results: {}\n".format(tempdir))
    
    # Save the data of each cell separately in the temporary directory as a pickle file
    for cellID in cte.data:
        filename = os.path.join(tempdir, '{}_data.pickle'.format(cellID))
        with open(filename, 'wb') as f:
            pickle.dump(cte.data[cellID], f)
    
    # set the parallel and reduce tasks
    parallel_task = partial(parallelization.homoprox_parallel, config=config, tempdir=tempdir)
    reduce_task = partial(parallelization.homoprox_reduce, tempdir=tempdir)
    
    # create a Controller
    controller = Controller(config)

    # run the parallel and reduce tasks
    homoprox_ratio = controller.map_reduce(parallel_task, reduce_task, args=list(cte.data.keys()))
    
    # Delete the non-empty temporary directory
    os.system('rm -r {}'.format(tempdir))
    
    return homoprox_ratio
