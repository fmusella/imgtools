import os
import sys
import pickle
import tempfile
from functools import partial
import typing
import numpy as np
import h5py
from alabtools.utils import Index
from alabtools.parallel import Controller


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
    
    # Add the index key if not present
    if not 'index' in required_keys:
        required_keys['index'] = {'type': bool}
    
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


def control_func(
    data: dict,
    data_attrs: dict,
    index: Index,
    config: dict,
    required_keys: dict,
    func_parallel: typing.Callable,  # ?
    reduce_initialization: typing.Callable,
    reduce_update: typing.Callable
):
    
    # Create a temporary directory
    tempdir = tempfile.mkdtemp(dir=os.getcwd())
    sys.stdout.write("Temporary directory for nodes' results: {}\n".format(tempdir))
    
    # Iterate over the cells and save the data in the temporary directory
    for cellID in data.keys():
        filename = os.path.join(tempdir, '{}_data.pickle'.format(cellID))
        with open(filename, 'wb') as f:
            pickle.dump(data[cellID], f)
    
    # Save the data attributes in the temporary directory
    with open(os.path.join(tempdir, 'data_attrs.pickle'), 'wb') as f:
        pickle.dump(data_attrs, f)
    
    # If the 'index' key is True, save the index in the temporary directory
    if config['index']:
        with h5py.File(os.path.join(tempdir, 'index.hdf5'), 'w') as f:
            index.save(f)
    
    # create a Controller
    controller = Controller(config)

    # run the parallel and reduce tasks
    parallel_task = partial(
        parallel_general,
        config=config,
        tempdir=tempdir,
        required_keys=required_keys,
        func_parallel=func_parallel  # ?
    )
    reduce_task = partial(
        reduce_general,
        config=config,
        tempdir=tempdir,
        reduce_initialization=reduce_initialization,
        reduce_update=reduce_update
    )
    result = controller.map_reduce(
        parallel_task,
        reduce_task,
        args = list(data.keys())
    )
    
    # Delete the non-empty temporary directory
    os.system('rm -r {}'.format(tempdir))
    
    del controller
    
    return result

def parallel_general(cellID, config: dict, tempdir: str, required_keys: dict, func_parallel: typing.Callable):
    
    # Check that the required keys are in the config
    check_config(config, required_keys)
    
    # Load the data for the cell with pickle
    in_filename = os.path.join(tempdir, '{}_data.pickle'.format(cellID))
    with open(in_filename, 'rb') as f:
        cell_data = pickle.load(f)
    
    # Load the data attributes
    with open(os.path.join(tempdir, 'data_attrs.pickle'), 'rb') as f:
        data_attrs = pickle.load(f)
    
    # Load the index, if required
    if config['index']:
        with h5py.File(os.path.join(tempdir, 'index.hdf5'), 'r') as f:
            index = Index(f)
    else:
        index = None
    
    # Perform the cell task on the data
    cell_result = func_parallel(cell_data, data_attrs, index, config)  # ?
    
    # Save the cell results in the temporary directory as a pickle file
    out_filename = os.path.join(tempdir, '{}_result.pickle'.format(cellID))
    with open(out_filename, 'wb') as f:
        pickle.dump(cell_result, f)
    
    del cell_data, cell_result
    
    return cellID

def reduce_general(
    cellIDs: list,
    config: dict,
    tempdir: str,
    reduce_initialization: typing.Callable,
    reduce_update: typing.Callable
):
    
    assert isinstance(cellIDs, list), "cellIDs should be a list. Got type: {}".format(type(cellIDs))
    assert len(cellIDs) > 0, "cellIDs should not be empty."
    
    # Load the data attributes
    with open(os.path.join(tempdir, 'data_attrs.pickle'), 'rb') as f:
        data_attrs = pickle.load(f)
    
    # Load the index, if required
    if config['index']:
        with h5py.File(os.path.join(tempdir, 'index.hdf5'), 'r') as f:
            index = Index(f)
    else:
        index = None
    
    # Initialize the result using the 'reduce_initialization' function
    result = reduce_initialization(cellIDs, data_attrs, index, config)
    
    # Iterate over the cellIDs and update the result using the 'reduce_update' function
    for cellID in cellIDs:
        
        # Get the filename for the temporary chromosomal volumes of the cell
        filename = os.path.join(tempdir, '{}_result.pickle'.format(cellID))
        assert os.path.isfile(filename), "Parallel result file for cell {} not found.".format(cellID)
        
        # Load the cell file
        with open(filename, 'rb') as f:
            cell_result = pickle.load(f)
        
        # Update the result
        result = reduce_update(cellID, result, cell_result, cellIDs, data_attrs, index, config)
    
    return result
