# Functions that process a ChromatinTracingExperiment object to get new data to be stored in the database

import numpy as np
from .cte import ChromatinTracingExperiment
from . import cte_utils
from . import parallelization
from . import metrics
from ..tracing import GenomicIterativeDBSCAN
from ..tracing import WardSpectralClustering
from .. import utils


# TRACING

def run_tracing(cte: ChromatinTracingExperiment, config: dict) -> ChromatinTracingExperiment:
    """ Performs the tracing on the population in parallel.

    Args:
        cte (ChromatinTracingExperiment)
        config (dict): configuration dictionary for the tracing task.

    Returns:
        cte_traced (ChromatinTracingExperiment): a new ChromatinTracingExperiment object with the traced data.
    """
    
    # Check that the tracing method is specified in the config and is valid
    if not 'method' in config:
        raise ValueError("Tracing method not specified in config.")
    if not config['method'] in tracing_available_methods:
        raise NotImplementedError("Tracing method {} not implemented.\nMust be one of: {}".
                                  format(config['method'],
                                         tracing_available_methods))
    
    def _rfunc_init(_1, _2, _3, _4, _5) -> dict:
        """ Initialize the traced data dictionary for the reduce function.

        Args:
            _*: not used, just to match the signature of the function

        Returns:
            traced_data (dict): empty dictionary
        """
        traced_data = {}
        return traced_data
    
    def _rfunc_update(cellID: str, traced_data: dict, cell_traced_data: dict, _2, _3, _4, _5, _6) -> dict:
        """ Update the traced data dictionary for the reduce function.

        Args:
            cellID (str)
            traced_data (dict): traced data dictionary of the entire population
            cell_traced_data (dict): traced data dictionary of a single cell
            _*: not used, just to match the signature of the function

        Returns:
            traced_data (dict): updated traced data dictionary of the entire population, with the data of cellID added
        """
        traced_data[cellID] = cell_traced_data
        return traced_data
    
    # Perform parallelization: get traced data
    cte_data_traced = parallelization.control_func(
        cte,
        config,
        tracing_required_keys[config['method']],
        _tracing_nfunc,
        _rfunc_init,
        _rfunc_update
    )
    
    # Initialize the traced CTE object and add the traced data
    cte_traced = ChromatinTracingExperiment()
    cte_traced.add_data(data=cte_data_traced, assembly=cte.assembly, index=cte.index)
    
    del cte_data_traced
    
    return cte_traced

def run_tracing_single_chrom(cte: ChromatinTracingExperiment, cellID: str, chrom: str, config: dict) -> ChromatinTracingExperiment:
    """Performs a tracing algorithm on a single chromosome of a single cell.

    Args:
        cellID (str): cell ID.
        chrom (str): chromosome.
        params (dict): configuration dictionary for the Genomic Iterative DBSCAN algorithm.
    
    Returns:
        cte_chrom_traced (ChromatinTracingExperiment): a new ChromatinTracingExperiment object with the traced data.
    """
    
    # Check that 'method' is specified in the config and is valid
    if not 'method' in config:
        raise ValueError("Tracing method not specified in config.")
    if not config['method'] in tracing_available_methods:
        raise NotImplementedError("Tracing method {} not implemented.\nMust be one of: {}".
                                  format(config['method'],
                                         tracing_available_methods))
    # Check that all required keys are present in params
    parallelization.check_config(config, tracing_required_keys[config['method']], parallel=False)
    
    # Perform the tracing
    traced_chrom_data = _chrom_tracing(chrom, cte.data[cellID][chrom], config)
    
    # Create a new ChromatinTracingExperiment object
    cte_chrom_traced = ChromatinTracingExperiment()
    
    # Add the traced data to the new ChromatinTracingExperiment object
    cte_chrom_traced.add_data(data={cellID: {chrom: traced_chrom_data}}, assembly=cte.assembly, index=cte.index)
    
    del traced_chrom_data
    
    return cte_chrom_traced

tracing_available_methods = ['gidbscan', 'wsclustering']
tracing_required_keys = {
    'gidbscan': {
        'dbscan_eps': {'type': float, 'positive': True},
        'dbscan_min_samples': {'type': int, 'positive': True},
        'window_size': {'type': int, 'positive': True},
        'delta': {'type': float, 'positive': True},
        'merging_proximity_length': {'type': int, 'positive': True},
        'merging_overlap_threshold': {'type': float, 'positive': True},
        'merging_distance_threshold': {'type': float, 'positive': True},
    },
    'wsclustering': {
        'n_clusters': {'type': int, 'positive': True},
        'st': {'type': float, 'positive': True},
        'ot': {'type': float, 'positive': True},
    }
}

def _tracing_nfunc(cell_data: dict, _1, _2, _3, config: dict) -> dict:
    """ Node function for the tracing task.

    Args:
        cell_data (dict): data of a single cell
        config (dict): configuration dictionary for the tracing task
        _*: not used, just to match the signature of the function

    Returns:
        cell_traced_data (dict): traced data of a single cell
    """
    
    # Initialize the traced data for the cell
    cell_data_traced = {}
    
    # Loop over chromosomes and perform tracing
    for chrom in cell_data:
        
        chrom_data = cell_data[chrom]
        chrom_data_traced = _chrom_tracing(chrom, chrom_data, config)
        cell_data_traced[chrom] = chrom_data_traced
        del chrom_data, chrom_data_traced
    
    return cell_data_traced

def _chrom_tracing(chrom: str, chrom_data: dict, params: dict):
    """Perform chromosome tracing on the data of a single chromosome.
    Returns the traced data of a single chromosome in dictionary format with new traceIDs.

    Args:
        chrom (str): The chromosome name.
        chrom_data (dict): The data of a single chromosome in dictionary format.
        params (dict): Parameters for GIDBSCAN.

    Returns:
        traced_chrom_data (dict): The traced data of a single chromosome in dictionary format.
    """
    
    # Convert the data to numpy arrays
    xs, ys, zs, starts, ends, lums, _, spotIDs = cte_utils.chrom_dict_to_numpy(chrom_data)
    coords = np.array([xs, ys, zs]).T
    
    # Perform tracing
    # GIDBSCAN
    if params['method'] == 'gidbscan':
        tracer = GenomicIterativeDBSCAN(
            params['dbscan_eps'],
            params['dbscan_min_samples'],
            params['window_size'],
            params['delta'],
            params['merging_proximity_length'],
            params['merging_overlap_threshold'],
            params['merging_distance_threshold']
            )
        tracer.fit(coords, starts)
    # Ward Spectral Clustering
    elif params['method'] == 'wsclustering':
        tracer = WardSpectralClustering(
            params['n_clusters'],
            params['st'],
            params['ot']
        )
        tracer.fit(coords)
    # Other methods
    else:
        raise NotImplementedError("Tracing method {} not implemented.".format(params['method']))
    
    # Get the traceIDs and convert them to strings
    traceIDs = tracer.labels_.astype('U10')
    
    # Convert the results back to dictionary format
    traced_chrom_data = cte_utils.chrom_numpy_to_dict(chrom, xs, ys, zs, starts, ends, lums, traceIDs, spotIDs)
    
    del xs, ys, zs, starts, ends, lums, spotIDs, coords, tracer, traceIDs
    
    return traced_chrom_data


# ALPHASHAPE

def run_alphashape(cte: ChromatinTracingExperiment, config: dict) -> dict:
    """ Performs the alphashape computation on the population in parallel.

    Args:
        cte (ChromatinTracingExperiment)
        config (dict): configuration dictionary for the alphashape task.

    Returns:
        alphashapes (dict): dictionary of the alphashapes of the population:
                            alphashapes[cellID] = {'alpha': alpha, 'mesh': mesh, 'volume': volume}
    """
    
    def _rfunc_init(_1, _2, _3, _4, _5) -> dict:
        """ Initialize the alphashapes dictionary for the reduce function.

        Args:
            _*: not used, just to match the signature of the function

        Returns:
            alphashapes (dict): empty dictionary
        """
        alphashapes = {}
        return alphashapes
    
    def _rfunc_update(cellID: str, alphashapes: dict, cell_alphamesh: dict, _2, _3, _4, _5, _6) -> dict:
        """ Update the alphashape dictionary for the reduce function.

        Args:
            cellID (str)
            alphashapes (dict): alphashapes dictionary of the entire population
            cell_alphamesh (dict): alpha-value and mesh of a single cell
            _*: not used, just to match the signature of the function

        Returns:
            alphashapes (dict): updated alphashapes dictionary of the entire population, with the data of cellID added
        """
        # Add the data of the cell to the alphashapes dictionary
        alphashapes[cellID] = {
            'alpha': cell_alphamesh['alpha'],
            'mesh': cell_alphamesh['mesh'],
            'volume': cell_alphamesh['mesh'].volume
        }
        return alphashapes
    
    alphashapes = parallelization.control_func(
        cte,
        config,
        alphashape_required_keys,
        _alphashape_nfunc,
        _rfunc_init,
        _rfunc_update
    )
    
    return alphashapes

def run_alphashape_single_cell(cte: ChromatinTracingExperiment, cellID: str, config: dict) -> tuple:
    """ Performs the alphashape computation on a single cell.

    Args:
        cte (ChromatinTracingExperiment)
        cellID (str)
        config (dict): configuration dictionary for the alphashape task.

    Returns:
        alpha (float): alpha parameter of the alphashape.
        mesh (trimesh.Trimesh): mesh of the alphashape.
    """
    
    # Check that all required keys are present in params
    parallelization.check_config(config, alphashape_required_keys, parallel=False)
    
    # Perform the alphashape computation
    cell_alphamesh = _alphashape_nfunc(cte.data[cellID], None, None, None, config)
    
    return cell_alphamesh['alpha'], cell_alphamesh['mesh']

alphashape_required_keys = {
        'alpha': {'type': float, 'positive': True},
        'force': {'type': bool}
}

def _alphashape_nfunc(cell_data: dict, _1, _2, _3, config: dict) -> dict:
    """ Node function for the alphashape task.
    It converts the data from dictionary to numpy format, fits the alphashape and returns the alpha value and the mesh.

    Args:
        cell_data (dict): data of a single cell
        config (dict): configuration dictionary for the alphashape task

    Returns:
        cell_alphamesh (dict): alpha value and mesh of a single cell
    """
    
    # Convert the data to numpy arrays
    xs, ys, zs, _, _, _, _, _, _ = cte_utils.cell_to_numpy(cell_data)
    points = np.array([xs, ys, zs]).T
    
    # Fit the alphashape
    alpha, mesh = utils.fit_alphashape(points, config['alpha'], config['force'])
    
    # Return the alpha value and the mesh
    cell_alphamesh = {'alpha': alpha, 'mesh': mesh}
    
    del xs, ys, zs, points, alpha, mesh
    
    return cell_alphamesh


# CLEANING

def run_cleaning(cte: ChromatinTracingExperiment, coverage_threshold: float, gendist_threshold: float) -> ChromatinTracingExperiment:
    """ Performs the cleaning of the traced data.
    
    Creates a new ChromatinTracingExperiment object with the cleaned data, i.e. without:
        - noisy traces
        - traces with a too-low coverage (less than coverage_threshold)
        - traces with a too-large minimum genomic distance between neighbors

    Args:
        cte (ChromatinTracingExperiment)
        coverage_threshold (float): minimum coverage for a trace to be kept.
        gendist_threshold (float): maximum threshold for the minimum genomic distance between neighbors for a trace to be kept.

    Returns:
        cte_clean (ChromatinTracingExperiment): a new ChromatinTracingExperiment object with the cleaned data.
    """
    
    # Initialize the cleaned data
    clean_data = {}
    
    # Loop over cells, chromosomes and traces and fill lists
    for cellID in cte.data:
        clean_data[cellID] = {}  # initialize dictionary for cellID
        
        for chrom in cte.data[cellID]:
            clean_data[cellID][chrom] = {}  # initialize dictionary for chrom
            
            for traceID in cte.data[cellID][chrom]:
                    
                    # Ignore noisy traces
                    if cte.look_for_noisy_trace(traceID):
                        continue
                    
                    # Ignore traces with low coverage
                    coverage = metrics.compute_trace_coverage(cte, cellID, chrom, traceID)
                    if coverage < coverage_threshold:
                        continue
                    
                    # Compute the minimum genomic distance between neighboring spots
                    gdist, _ = metrics.compute_trace_neighbor_distances(cte, cellID, chrom, traceID)
                    min_gdist = np.min(gdist)
                    if min_gdist > gendist_threshold:
                        continue
                    
                    # If everything is ok, add the trace to the cleaned data
                    clean_data[cellID][chrom][traceID] = cte.data[cellID][chrom][traceID]
            
            # If chrom data is empty, delete it
            if clean_data[cellID][chrom] == {}:
                del clean_data[cellID][chrom]
        
        # If cell data is empty, delete it
        if clean_data[cellID] == {}:
            del clean_data[cellID]
    
    # Create a new ChromatinTracingExperiment object
    cte_clean = ChromatinTracingExperiment()
    
    # Add the traced data to the new ChromatinTracingExperiment object
    cte_clean.add_data(data=clean_data, assembly=cte.assembly, index=cte.index)
    
    del clean_data
    
    return cte_clean


# TRIMMING

def trim_trace_data(cte: ChromatinTracingExperiment, cellID: str, chrom: str, traceID: str) -> dict:
    """ Remove multiple spots associated with the same domain in a trace.
    
    It uses the spots_3d_median function to choose a spot among the repeated ones:
        - If there are two spots, it chooses the one with closest distance to the trace's Center of Mass.
        - If there are more than two spots, it chooses the one with minimum average distance to the other spots.

    Args:
        cte (ChromatinTracingExperiment)
        cellID (str)
        chrom (str)
        traceID (str)
    
    Returns:
        trimmed_trace_data (dict): dictionary of the trimmed trace data.
    """
            
    # Take the trace data
    try:
        trace_data = cte.data[cellID][chrom][traceID]
    except KeyError:
        raise KeyError("CellID {}, chrom {} and traceID {} not in data.".format(cellID, chrom, traceID))
    
    # Convert the trace data to numpy array format
    xs, ys, zs, chroms, starts, ends, lums, spotIDs = cte_utils.trace_dict_to_numpy(trace_data)
    
    # Compute the Center of Mass of the trace
    com = np.array([np.mean(xs), np.mean(ys), np.mean(zs)])
    
    # Identify the domains as the (start, end) pairs (chrom is the same for all spots in the trace)
    domains = np.array([starts, ends]).T
    
    # Identify the unique domains
    unique_domains = np.unique(domains, axis=0)
    
    # If there are no repeated domains, return the original trace data
    if np.array_equal(domains, unique_domains):
        return trace_data
    
    # If there are repeated domains, trim them according to the 3D median procedure
    
    # Initialize the trimmed trace data
    trimmed_trace_data = {}
    
    for domain in unique_domains:
        
        # Find the indices associated with the domain
        indices = np.where(np.all(domains == domain, axis=1))[0]
        
        # Get the coordinates of the spots associated with the domain
        points = np.array([xs[indices], ys[indices], zs[indices]]).T
        
        # Compute the spots 3D median, getting the index - among points - of the 3D median spot
        median_idx = utils.spots_3d_median(points, com)
        
        # Get the index of the median spot in the indices array
        median_idx = indices[median_idx]
        
        assert median_idx in indices, "Median index not in indices. Something went wrong."
        
        trimmed_trace_data[spotIDs[median_idx]] = {
                'x': float(xs[median_idx]),
                'y': float(ys[median_idx]),
                'z': float(zs[median_idx]),
                'chrom': str(chroms[median_idx]),
                'start': int(starts[median_idx]),
                'end': int(ends[median_idx]),
                'lum': float(lums[median_idx])
            }
    
    return trimmed_trace_data

def run_trim(cte: ChromatinTracingExperiment) -> ChromatinTracingExperiment:
    """ Trim the data, removing multiple spots associated with the same domain in each trace.
    
    Args:
        cte (ChromatinTracingExperiment)

    Returns:
        cte_trimmed (ChromatinTracingExperiment): a new ChromatinTracingExperiment object with the trimmed data.
    """
    
    trimmed_data = {}
    
    # Loop over cells, chromosomes and traces and trim the trace data
    for cellID in cte.data:
        if cellID not in trimmed_data:
            trimmed_data[cellID] = {}
        for chrom in cte.data[cellID]:
            if chrom not in trimmed_data[cellID]:
                trimmed_data[cellID][chrom] = {}
            for traceID in cte.data[cellID][chrom]:
                trimmed_data[cellID][chrom][traceID] = trim_trace_data(cte, cellID, chrom, traceID)
                
    # Create a new ChromatinTracingExperiment object
    cte_trimmed = ChromatinTracingExperiment()
    
    # Add the traced data to the new ChromatinTracingExperiment object
    cte_trimmed.add_data(data=trimmed_data, assembly=cte.assembly, index=cte.index)
    
    del trimmed_data
    
    return cte_trimmed
