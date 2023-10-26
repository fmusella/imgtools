import os
import pickle
import numpy as np
from .gi_dbscan import GenomicIterativeDBSCAN
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
