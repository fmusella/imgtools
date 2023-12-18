import os
import pickle
import numpy as np
from .gi_dbscan import GenomicIterativeDBSCAN
from .ward_spectral import WardSpectralClustering
import alphashape
import trimesh
from . import utils
from . import visualization


def check_config(config: dict, required_keys: dict, parallel: bool = True):
    """ Generic function for checking the config file for the parallelization tasks.

    Args:
        config (dict): config file for the parallelization tasks.
        required_keys (dict): dictionary of required keys for the config file.
    """
    
    if not isinstance(config, dict):
        raise TypeError("config should be a dictionary. Got type: {}".format(type(config)))
    
    if not isinstance(required_keys, dict):
        raise TypeError("required_keys should be a dictionary. Got type: {}".format(type(required_keys)))
    
    # Add the parallel key if parallel is True
    if parallel:
        required_keys['parallel'] = {'type': dict}
    
    for key in required_keys:
        # Check if the key is in the config
        if not key in config:
            raise ValueError("Key {} not found in config.".format(key))
        # Check if the type of the key is correct
        if not isinstance(config[key], required_keys[key]['type']):
            raise TypeError("Invalid type for key: {}. Got type: {}. Expected type: {}".format(key, type(config[key]), required_keys[key]['type']))
        # Check if numeric keys are positive
        if 'positive' in required_keys[key]:
            if not config[key] > 0:
                raise ValueError("Key {} should be positive. Got: {}".format(key, config[key]))


# PARALLEL FUNCTIONS FOR THE TRACING TASK

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


# PARALLEL FUNCTIONS FOR THE ALPHASHAPE TASK

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


# PARALLEL FUNCTIONS FOR THE MRC FILE GENERATION TASK

required_keys_mrc = {
    'resolution': {'type': float, 'positive': True},
    'border': {'type': int, 'positive': True},
    'surface_thickness': {'type': float, 'positive': True},
    'mrc_path': {'type': str},
}

def do_cell_mrc(cellID: str, cell_mesh: trimesh.Trimesh, params: dict):
    """ Generate the mrc file for a single cell.

    Args:
        cellID (str): The cell ID.
        cell_mesh (trimesh.Trimesh): The alphashape of the cell.
        params (dict): Parameters for the mrc file generation task.

    Returns:
        origin (tuple): Origin of the mrc file in voxel units.
        shape (tuple): Shape of the mrc file in voxel
    """
    
    origin, shape = visualization.mesh_to_mrc(
        path=params['mrc_path'],
        name_prefix=cellID,
        mesh=cell_mesh,
        resolution=params['resolution'],
        border=params['border'],
        surface_thickness=params['surface_thickness']
    )
    
    return origin, shape

def mrc_parallel(cellID: str, config: dict, tempdir: str):
    """Parallel function for the alphashape task.

    Args:
        cellID (str): The cell ID.
        config (dict): The config file for the alphashape task.
        tempdir (str): Temporary directory for storing intermediate results.
    """
    
    check_config(config, required_keys_mrc)
    
    assert isinstance(cellID, str), "cellID {} should be a string. Got type: {}".format(cellID, type(cellID))
    
    assert isinstance(tempdir, str), "tempdir should be a string. Got type: {}".format(type(tempdir))
    assert os.path.isdir(tempdir), "tempdir is not a valid directory."
    
    # Load file with the alphashape for the cell
    in_filename = os.path.join(tempdir, '{}_mesh.pickle'.format(cellID))
    
    assert os.path.isfile(in_filename), "Mesh for cell {} not found.".format(cellID)
    
    with open(in_filename, 'rb') as f:
        cell_mesh = pickle.load(f)
    
    # Write the mrc file for the cell
    origin, shape = do_cell_mrc(cellID, cell_mesh, config)
    
    del cell_mesh
    
    # Save the origin and shape for the cell with pickle
    out_filename = os.path.join(tempdir, '{}_mrc_params.pickle'.format(cellID))
    with open(out_filename, 'wb') as f:
        pickle.dump({'origin': origin, 'shape': shape}, f)
    
    return cellID

def mrc_reduce(cellIDs: list, config: dict, tempdir: str):
    """ Reduce function for the mrc file generation task.
    
    Collects the parameters of the mrc files for all cells.

    Args:
        cellIDs (list): list of cell IDs.
        tempdir (str): temporary directory for storing intermediate results.

    Returns:
        mrc_params (dict): Dictionary of mrc parameters for all cells in dictionary format.
    """
    
    # Check cellIDs
    assert isinstance(cellIDs, list), "cellIDs should be a list. Got type: {}".format(type(cellIDs))
    assert len(cellIDs) > 0, "cellIDs should not be empty."
    
    # Initialize the output, which is a dictionary of parameters for the mrc files
    mrc_params = {}

    for cellID in cellIDs:
        
        # Get the filename for the mrc parameters of the cell
        filename = os.path.join(tempdir, '{}_mrc_params.pickle'.format(cellID))
        
        # Check that the file exists
        assert os.path.isfile(filename), "MRC param file for cell {} not found.".format(cellID)

        # Load the file
        with open(filename, 'rb') as f:
            cell_mrc_params = pickle.load(f)
        
        # Get the data from the pickle file
        origin = cell_mrc_params['origin']
        shape = cell_mrc_params['shape']
        
        # Add the data to the output
        mrc_params[cellID] = {
            'origin': origin,
            'shape': shape
        }
    
    # Save the mrc parameters for all cells with pickle
    out_filename = os.path.join(config['mrc_path'], 'mrc_params.pickle')
    with open(out_filename, 'wb') as f:
        pickle.dump(mrc_params, f)
