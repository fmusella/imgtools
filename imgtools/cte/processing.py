# Functions that process a ChromatinTracingExperiment object to get new data to be stored in the database

import numpy as np
import trimesh
import alphashape
from .cte import ChromatinTracingExperiment
from . import utils
from ..tracing import GenomicIterativeDBSCAN
from ..tracing import WardSpectralClustering


# TRACING

acceptable_tracing_methods = ['gidbscan', 'wsclustering']

required_keys_tracing = {
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

def do_chromosome_tracing(chrom: str, chrom_data: dict, params: dict):
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
    xs, ys, zs, starts, ends, lums, _, spotIDs = utils.chrom_dict_to_numpy(chrom_data)
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
    traced_chrom_data = utils.chrom_numpy_to_dict(chrom, xs, ys, zs, starts, ends, lums, traceIDs, spotIDs)
    
    del xs, ys, zs, starts, ends, lums, spotIDs, coords, tracer, traceIDs
    
    return traced_chrom_data

def tracing_parallel(cellID: str, config: dict, tempdir: str):
    """Parallel function for the tracing task.

    Args:
        cellID (str): The cell ID.
        config (dict): The config file for the tracing task.
        tempdir (str): Temporary directory for storing intermediate results.

    Returns:
        cellID (str): The cell ID.
    """
    
    # Check that the tracing method is specified in the config, is valid and that the its parameters are given
    if not 'method' in config:
        raise ValueError("Tracing method not specified in config.")
    if not config['method'] in acceptable_tracing_methods:
        raise NotImplementedError("Tracing method {} not implemented. Must be one of: {}".format(config['method'],
                                                                                                 acceptable_tracing_methods))
    check_config(config, required_keys_tracing[config['method']])
    
    assert isinstance(cellID, str), "cellID should be a string. Got type: {}".format(type(cellID))
    
    assert isinstance(tempdir, str), "tempdir should be a string. Got type: {}".format(type(tempdir))
    assert os.path.isdir(tempdir), "tempdir should be a directory. Got: {}".format(tempdir)
    
    # Try to load the data for the cell with pickle
    in_filename = os.path.join(tempdir, '{}_data.pickle'.format(cellID))
    assert os.path.isfile(in_filename), "Data for cell {} not found.".format(cellID)
    with open(in_filename, 'rb') as f:
        cell_data = pickle.load(f)
    
    # Initialized traced data for the cell
    traced_cell_data = {}
    
    # Perform tracing on each chromosome
    for chrom in cell_data:
                
        traced_chrom_data = do_chromosome_tracing(chrom, cell_data[chrom], config)
        traced_cell_data[chrom] = traced_chrom_data
        del traced_chrom_data
    
    # Save the traced data for the cell with pickle
    out_filename = os.path.join(tempdir, '{}_traced_data.pickle'.format(cellID))
    with open(out_filename, 'wb') as f:
        pickle.dump(traced_cell_data, f)
    
    del cell_data, traced_cell_data
    
    return cellID

def tracing_reduce(cellIDs: list, tempdir: str):
    """Reduce function for the tracing task.
    
    Takes the traced cell data from the parallel function and combines them into a single dictionary.

    Args:
        cellIDs (list): List of cell IDs.
        config (dict): Config file for the tracing task.
        tempdir (str): Temporary directory for storing intermediate results.

    Returns:
        traced_data (dict): The traced data for all cells in dictionary format.
    """
    
    assert isinstance(cellIDs, list), "cellIDs should be a list. Got type: {}".format(type(cellIDs))
    assert len(cellIDs) > 0, "cellIDs should not be empty."
    
    traced_data = {}

    for cellID in cellIDs:
        
        # Get the filename for the temporary traced data for the cell
        filename = os.path.join(tempdir, '{}_traced_data.pickle'.format(cellID))
        
        assert os.path.isfile(filename), "Traced data for cell {} not found.".format(cellID)

        with open(filename, 'rb') as f:
            cell_data = pickle.load(f)
        
        traced_data[cellID] = cell_data
    
    return traced_data

def run_tracing(cte: ChromatinTracingExperiment, config: dict) -> ChromatinTracingExperiment:
    """ Performs a tracing algorithm on the population.
    
    Accepts either serial or parallel computation, as specified by the alabtools.parallel.Controller class.

    Args:
        config (dict): configuration dictionary for the Genomic Iterative DBSCAN algorithm.

    Returns:
        cte_traced (ChromatinTracingExperiment): a new ChromatinTracingExperiment object with the traced data.
    """
    
    # Create a temporary directory
    tempdir = tempfile.mkdtemp(dir=os.getcwd())
    sys.stdout.write("Temporary directory for nodes' results: {}\n".format(tempdir))
    
    # Save the data of each cell separately in the temporary directory as a pickle file
    for cellID in cte.data:
        filename = os.path.join(tempdir, '{}_data.pickle'.format(cellID))
        with open(filename, 'wb') as f:
            pickle.dump(cte.data[cellID], f)
    
    # set the parallel and reduce tasks
    parallel_task = partial(tracing_parallel, config=config, tempdir=tempdir)
    reduce_task = partial(tracing_reduce, tempdir=tempdir)
    
    # create a Controller
    controller = Controller(config)

    # run the parallel and reduce tasks
    traced_data = controller.map_reduce(parallel_task, reduce_task, args=list(cte.data.keys()))
    
    # Delete the non-empty temporary directory
    os.system('rm -r {}'.format(tempdir))
    
    # Create a new ChromatinTracingExperiment object
    cte_traced = ChromatinTracingExperiment()

    # Add the traced data to the new ChromatinTracingExperiment object
    cte_traced.add_data(data=traced_data, assembly=cte.assembly, index=cte.index)
    
    del controller, traced_data
    
    return cte_traced


def do_tracing_single_chrom(cte: ChromatinTracingExperiment, cellID: str, chrom: str, params: dict) -> ChromatinTracingExperiment:
    """Performs a tracing algorithm on a single chromosome of a single cell.

    Args:
        cellID (str): cell ID.
        chrom (str): chromosome.
        params (dict): configuration dictionary for the Genomic Iterative DBSCAN algorithm.
    
    Returns:
        cte_chrom_traced (ChromatinTracingExperiment): a new ChromatinTracingExperiment object with the traced data.
    """
    
    # Check that all required keys are present in params
    if 'method' not in params:
        raise ValueError("params must contain a 'method' key.")
    if params['method'] not in acceptable_tracing_methods:
        raise ValueError("Method {} not recognized. Must be one of {}.".format(params['method'], acceptable_tracing_methods))
    check_config(params, required_keys_tracing[params['method']], parallel=False)
    
    # Perform the tracing
    traced_chrom_data = do_chromosome_tracing(chrom, cte.data[cellID][chrom], params)
    
    # Create a new ChromatinTracingExperiment object
    cte_chrom_traced = ChromatinTracingExperiment()
    
    # Add the traced data to the new ChromatinTracingExperiment object
    cte_chrom_traced.add_data(data={cellID: {chrom: traced_chrom_data}}, assembly=cte.assembly, index=cte.index)
    
    del traced_chrom_data
    
    return cte_chrom_traced


# ALPHASHAPE

required_keys_alphashape = {
        'alpha': {'type': float, 'positive': True},
        'force': {'type': bool}
}

def do_cell_alphashape(cell_data: dict, params: dict):
    """
    Fits an alpha-shape to contain all the points in the cell.
    
    If force is True, the alpha-shape is fitted with the input alpha value.
    
    Otherwise, the alpha value is found by a search algorithm starting from the input one
    and halving it until a closed alpha-shape is found.
    A hard-coded maximum number of iterations is used to avoid infinite loops.
    
    Args:
        cell_data (dict): The data of a single cell in dictionary format.
        params (dict): Parameters for the alphashape task.
    
    Returns:
        alpha (float): alpha value used to fit the alpha-shape.
        mesh (trimesh.Trimesh): alpha-shape fitted to the input points.
    """
    
    # Convert the data to numpy arrays
    xs, ys, zs, _, _, _, _, _, _ = utils.cell_to_numpy(cell_data)
    points = np.array([xs, ys, zs]).T
    
    # The alphashape code doesn't give closed shapes if the input points are not float64
    points = points.astype(np.float64)
    
    # If force, we only use the input alpha value
    if params['force']:
        alpha_shape = alphashape.alphashape(points, params['alpha'])
        mesh = trimesh.Trimesh(vertices=alpha_shape.vertices, faces=alpha_shape.faces, process=True)
        if not mesh.is_watertight:
            raise ValueError("The alpha-shape is not closed with the input alpha value forced. Try setting force=False.")
        return alpha, mesh
    
    # If not force, we find the alpha value by a search algorithm,
    # where we start with the input alpha and - if the shape is not closed - we halve it.
    max_iter = 20
    counter = 0
    alpha = params['alpha']
    while True:
        counter += 1
        if counter > max_iter:
            raise ValueError("Maximum number of iterations reached, but no closed alpha-shape found.")
        alpha_shape = alphashape.alphashape(points, alpha)
        mesh = trimesh.Trimesh(vertices=alpha_shape.vertices, faces=alpha_shape.faces, process=True)
        if mesh.is_watertight:
            break
        alpha = alpha / 2
    
    return alpha, mesh

def alphashape_parallel(cellID: str, config: dict, tempdir: str):
    """Parallel function for the alphashape task.

    Args:
        cellID (str): The cell ID.
        config (dict): The config file for the alphashape task.
        tempdir (str): Temporary directory for storing intermediate results.

    Returns:
        cellID (str): The cell ID.
    """
    
    # Check the cellID
    if not isinstance(cellID, str):
        raise TypeError("cellID {} should be a string. Got type: {}".format(cellID, type(cellID)))
    
    # Check the config file
    check_config(config, required_keys_alphashape)
    
    # Check that the tempdir is valid
    if not isinstance(tempdir, str):
        raise TypeError("tempdir should be a string. Got type: {}".format(type(tempdir)))
    if not os.path.isdir(tempdir):
        raise NotADirectoryError("tempdir is not a valid directory.")
    
    # Try to load the data for the cell with pickle
    in_filename = os.path.join(tempdir, '{}_data.pickle'.format(cellID))
    if not os.path.isfile(in_filename):
        raise FileNotFoundError("Data for cell {} not found.".format(cellID))
    try:
        with open(in_filename, 'rb') as f:
            cell_data = pickle.load(f)
    except:
        raise ValueError("Data for cell {} is not a valid pickle file.".format(cellID))
    
    # Get the alphashape for the cell
    alpha, mesh = do_cell_alphashape(cell_data, config)
    
    # Save the alphashape for the cell with pickle
    out_filename = os.path.join(tempdir, '{}_alphamesh.pickle'.format(cellID))
    with open(out_filename, 'wb') as f:
        pickle.dump({'alpha': alpha, 'mesh': mesh}, f)
    
    del cell_data, mesh
    
    return cellID
    
def alphashape_reduce(cellIDs: list, tempdir: str):
    """ Reduce function for the alphashape task.

    Args:
        cellIDs (list): List of cell IDs.
        tempdir (str): Temporary directory for storing intermediate results.
    
    Returns:
        alphashapes (dict): Dictionary of alphashapes for all cells in dictionary format.
                            Keys are cellIDs, values are dictionaries with keys 'alpha', 'mesh' and 'volume'.
    """
    
    # Check cellIDs
    assert isinstance(cellIDs, list), "cellIDs should be a list. Got type: {}".format(type(cellIDs))
    assert len(cellIDs) > 0, "cellIDs should not be empty."
    
    # Initialize the output, which is a dictionary of alphashapes
    alphashapes = {}

    for cellID in cellIDs:
        
        # Get the filename for the temporary alphashape of the cell
        filename = os.path.join(tempdir, '{}_alphamesh.pickle'.format(cellID))
        
        # Check that the file exists
        assert os.path.isfile(filename), "Alphashape file for cell {} not found.".format(cellID)

        # Load the alphashape
        try:
            with open(filename, 'rb') as f:
                cell_alphamesh = pickle.load(f)
        except:
            raise ValueError("Alphashape file for cell {} is not a valid pickle file.".format(cellID))
        
        # Get the data from the pickle file
        alpha_cell = cell_alphamesh['alpha']
        mesh_cell = cell_alphamesh['mesh']
        volume_cell = mesh_cell.volume
        del cell_alphamesh
        
        # Add the data to the output
        alphashapes[cellID] = {
            'alpha': alpha_cell,
            'mesh': mesh_cell,
            'volume': volume_cell
        }
    
    return alphashapes

def run_alphashape(cte: ChromatinTracingExperiment, config: dict) -> None:
    """ Performs the alphashape computation on the population.

    Args:
        config (dict): configuration dictionary for the alphashape computation.
    """
    
    # Create a temporary directory
    tempdir = tempfile.mkdtemp(dir=os.getcwd())
    sys.stdout.write("Temporary directory for nodes' results: {}\n".format(tempdir))
    
    # Save the data of each cell separately in the temporary directory as a pickle file
    for cellID in cte.data:
        filename = os.path.join(tempdir, '{}_data.pickle'.format(cellID))
        with open(filename, 'wb') as f:
            pickle.dump(cte.data[cellID], f)
    
    # set the parallel and reduce tasks
    parallel_task = partial(parallelization.alphashape_parallel, config=config, tempdir=tempdir)
    reduce_task = partial(parallelization.alphashape_reduce, tempdir=tempdir)
    
    # create a Controller
    controller = Controller(config)

    # run the parallel and reduce tasks
    alphashapes = controller.map_reduce(parallel_task, reduce_task, args=list(cte.data.keys()))
    
    # Delete the non-empty temporary directory
    os.system('rm -r {}'.format(tempdir))
    
    # Store the alphashape in the ChromatinTracingExperiment object
    cte.alphashapes = alphashapes
    
    del controller, alphashapes

def run_alphashape_single_cell(cte: ChromatinTracingExperiment, cellID: str, params: dict) -> (float, trimesh.Trimesh):
    """ Performs the alphashape computation on a single cell.

    Args:
        cellID (str): cell ID.
        params (dict): configuration dictionary for the alphashape computation.

    Returns:
        alpha (float): alpha parameter of the alphashape.
        mesh (trimesh.Trimesh): mesh of the alphashape.
    """
    
    # Check that all required keys are present in params
    parallelization.check_config(params, parallelization.required_keys_alphashape, parallel=False)
    
    # Perform the alphashape computation
    alpha, mesh = parallelization.do_cell_alphashape(cte.data[cellID], params)
    
    return alpha, mesh


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
                    coverage = cte.compute_trace_coverage(cellID, chrom, traceID)
                    if coverage < coverage_threshold:
                        continue
                    
                    # Compute the minimum genomic distance between neighboring spots
                    gdist, _ = cte.compute_trace_neighbor_distances(cellID, chrom, traceID)
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
    xs, ys, zs, chroms, starts, ends, lums, spotIDs = utils.trace_dict_to_numpy(trace_data)
    
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
                trimmed_data[cellID][chrom][traceID] = trim_trace_data(cellID, chrom, traceID)
                
    # Create a new ChromatinTracingExperiment object
    cte_trimmed = ChromatinTracingExperiment()
    
    # Add the traced data to the new ChromatinTracingExperiment object
    cte_trimmed.add_data(data=trimmed_data, assembly=cte.assembly, index=cte.index)
    
    del trimmed_data
    
    return cte_trimmed

