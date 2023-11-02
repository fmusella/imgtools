import os
import pickle
import numpy as np
from .gi_dbscan import GenomicIterativeDBSCAN
import alphashape
import trimesh
from . import utils


# PARALLEL FUNCTIONS FOR THE TRACING TASK

def check_config_tracing(config: dict, parallel: bool = True):
    """Check the config file for the tracing task.
    Raises an error if the config file is invalid.

    Args:
        config (dict): The config file for the tracing task.
    """
    
    assert isinstance(config, dict), "config should be a dictionary. Got type: {}".format(type(config))
    
    required_keys = [
        ('dbscan_eps', float),
        ('dbscan_min_samples', int),
        ('window_size', int),
        ('delta', float),
        ('merging_proximity_length', int),
        ('merging_overlap_threshold', float),
        ('merging_distance_threshold', float)
    ]
    
    # Add the parallel key if parallel is True
    if parallel:
        required_keys.append(('parallel', dict))
    
    for (key, type) in required_keys:
        assert key in config
        assert isinstance(config[key], type), "Invalid type for key: {}. Got type: {}. Expected type: {}".format(key, type(config[key]), type)
        if type == int or type == float:
            assert config[key] > 0, "Key {} should be positive. Got: {}".format(key, config[key])

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
    
    # Perform GIDBSCAN
    coords = np.array([xs, ys, zs]).T
    gidbscan = GenomicIterativeDBSCAN(
        params['dbscan_eps'],
        params['dbscan_min_samples'],
        params['window_size'],
        params['delta'],
        params['merging_proximity_length'],
        params['merging_overlap_threshold'],
        params['merging_distance_threshold']
        )
    
    gidbscan.fit(coords, starts)
    traceIDs = gidbscan.labels_.astype('U10')
    
    # Convert the results back to dictionary format
    traced_chrom_data = utils.chrom_numpy_to_dict(chrom, xs, ys, zs, starts, ends, lums, traceIDs, spotIDs)
    
    del xs, ys, zs, starts, ends, lums, spotIDs, coords, gidbscan, traceIDs
    
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
    
    check_config_tracing(config)
    
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
    
    assert isinstance(tempdir, str), "tempdir should be a string. Got type: {}".format(type(tempdir))
    assert os.path.isdir(tempdir), "tempdir should be a directory. Got: {}".format(tempdir)
    
    traced_data = {}

    for cellID in cellIDs:
        
        assert isinstance(cellID, str), "cellID should be a string. Got type: {}".format(type(cellID))
        
        # Get the filename for the temporary traced data for the cell
        filename = os.path.join(tempdir, '{}_traced_data.pickle'.format(cellID))
        
        assert os.path.isfile(filename), "Traced data for cell {} not found.".format(cellID)

        with open(filename, 'rb') as f:
            cell_data = pickle.load(f)
        
        traced_data[cellID] = cell_data
    
    return traced_data


# PARALLEL FUNCTIONS FOR THE ALPHASHAPE TASK

def check_config_alphashape(config: dict, parallel: bool = True):
    """ Check the config file for the alphashape task.

    Args:
        config (dict): The config file for the alphashape task.
        parallel (bool, optional): If True, the config file should contain a parallel key. Defaults to True.
    """
    
    if not isinstance(config, dict):
        raise TypeError("config should be a dictionary. Got type: {}".format(type(config)))
    
    required_keys = [
        ('alpha', float),
        ('force', bool)
    ]
    
    # Add the parallel key if parallel is True
    if parallel:
        required_keys.append(('parallel', dict))
    
    for (key, type) in required_keys:
        # Check if the key is in the config
        if not key in config:
            raise ValueError("Key {} not found in config.".format(key))
        # Check if the type of the key is correct
        if not isinstance(config[key], type):
            raise TypeError("Invalid type for key: {}. Got type: {}. Expected type: {}".format(key, type(config[key]), type))
        # Check if numeric keys are positive
        if type == int or type == float:
            if not config[key] > 0:
                raise ValueError("Key {} should be positive. Got: {}".format(key, config[key]))

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
    check_config_alphashape(config)
    
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
    if not isinstance(cellIDs, list):
        raise TypeError("cellIDs should be a list. Got type: {}".format(type(cellIDs)))
    if len(cellIDs) == 0:
        raise ValueError("cellIDs should not be empty.")
    
    # Check tempdir
    if not isinstance(tempdir, str):
        raise TypeError("tempdir should be a string. Got type: {}".format(type(tempdir)))
    if not os.path.isdir(tempdir):
        raise NotADirectoryError("tempdir is not a valid directory.")
    
    # Initialize the output, which is a dictionary of alphashapes
    alphashapes = {}

    for cellID in cellIDs:
        
        # Get the filename for the temporary alphashape of the cell
        filename = os.path.join(tempdir, '{}_alphamesh.pickle'.format(cellID))
        
        # Check that the file exists
        if not os.path.isfile(filename):
            raise FileNotFoundError("Alphashape file for cell {} not found.".format(cellID))

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
