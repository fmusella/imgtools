import os
import sys
import pickle
import tempfile
from functools import partial
import typing
import h5py
from alabtools.utils import Index
from alabtools.parallel import Controller
from .cte import ChromatinTracingExperiment


def check_config(config: dict, required_keys: dict, parallel: bool = True) -> None:
    """ Generic function for checking the config file for the parallelization tasks.

    Args:
        config (dict): config file for the parallelization tasks.
        required_keys (dict): dictionary of required keys for the config file.
        parallel (bool, optional): whether the parallelization is performed. Defaults to True.
    """
    
    if not isinstance(config, dict):
        raise TypeError("config should be a dictionary. Got type: {}".format(type(config)))
    
    if not isinstance(required_keys, dict):
        raise TypeError("required_keys should be a dictionary. Got type: {}".format(type(required_keys)))
    
    # Add the use_index key if not present
    if not 'use_index' in required_keys:
        required_keys['use_index'] = {'type': bool}
        
    # Add the use_alphashapes key if not present
    if not 'use_alphashapes' in required_keys:
        required_keys['use_alphashapes'] = {'type': bool}
    
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
    cte: ChromatinTracingExperiment,
    config: dict,
    required_keys: dict,
    func_parallel: typing.Callable,
    reduce_initialization: typing.Callable,
    reduce_update: typing.Callable
) -> object:
    """Generic function for controlling a parallelization task on a ChromatinTracingExperiment.

    Args:
        cte (ChromatinTracingExperiment)
        config (dict): config file for the parallelization tasks.
        required_keys (dict): required keys for the config file.
        func_parallel (typing.Callable): parallel function to be executed on each cell.
        reduce_initialization (typing.Callable): function to initialize the result object in the reduce task.
        reduce_update (typing.Callable): function to update the result object - with the cell results - in the reduce task.

    Returns:
        result (object): result of the parallelization task. Can be any object.
    """
    
    # Create a temporary directory
    tempdir = tempfile.mkdtemp(dir=os.getcwd())
    sys.stdout.write("Temporary directory for nodes' results: {}\n".format(tempdir))
    
    # Iterate over the cells and save the data in the temporary directory
    for cellID in cte.data.keys():
        filename = os.path.join(tempdir, '{}_data.pickle'.format(cellID))
        with open(filename, 'wb') as f:
            pickle.dump(cte.data[cellID], f)
    
    # Save the data attributes in the temporary directory
    with open(os.path.join(tempdir, 'data_attrs.pickle'), 'wb') as f:
        pickle.dump(cte.attrs, f)
    
    # If the 'use_index' key is True, save the index in the temporary directory
    if config['use_index']:
        with h5py.File(os.path.join(tempdir, 'index.hdf5'), 'w') as f:
            cte.index.save(f)
    
    # If the 'use_alphashapes' key is True, save the shape in the temporary directory
    if config['use_alphashapes']:
        with open(os.path.join(tempdir, 'alphashapes.pickle'), 'wb') as f:
            pickle.dump(cte.alphashapes, f)
    
    # create a Controller
    controller = Controller(config)

    # run the parallel and reduce tasks
    parallel_task = partial(
        parallel_general,
        config=config,
        tempdir=tempdir,
        required_keys=required_keys,
        func_parallel=func_parallel
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
        args = list(cte.data.keys())
    )
    
    # Delete the non-empty temporary directory
    os.system('rm -r {}'.format(tempdir))
    
    del controller
    
    return result

def parallel_general(
    cellID: str,
    config: dict,
    tempdir: str,
    required_keys: dict,
    func_parallel: typing.Callable
) -> str:
    """ Generic function for performing a parallelization task on a single cell.

    Args:
        cellID (str)
        config (dict)
        tempdir (str)
        required_keys (dict)
        func_parallel (typing.Callable)

    Returns:
        cellID (str)
    """
    
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
    if config['use_index']:
        with h5py.File(os.path.join(tempdir, 'index.hdf5'), 'r') as f:
            index = Index(f)
    else:
        index = None
    
    # Load the alphashapes, if required
    if config['use_alphashapes']:
        with open(os.path.join(tempdir, 'alphashapes.pickle'), 'rb') as f:
            alphashapes = pickle.load(f)
    else:
        alphashapes = None
    
    # Perform the cell task on the data
    cell_result = func_parallel(cell_data, data_attrs, index, alphashapes, config)
    
    # Save the cell results in the temporary directory as a pickle file
    out_filename = os.path.join(tempdir, '{}_result.pickle'.format(cellID))
    with open(out_filename, 'wb') as f:
        pickle.dump(cell_result, f)
    
    del cell_data, cell_result, data_attrs, index, alphashapes
    
    return cellID

def reduce_general(
    cellIDs: list,
    config: dict,
    tempdir: str,
    reduce_initialization: typing.Callable,
    reduce_update: typing.Callable
) -> object:
    """ Generic function for reducing the results of the parallelization task.

    Args:
        cellIDs (list)
        config (dict)
        tempdir (str)
        reduce_initialization (typing.Callable)
        reduce_update (typing.Callable)

    Returns:
        result (object)
    """
    
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
    
    # Load the alphashapes, if required
    if config['use_alphashapes']:
        with open(os.path.join(tempdir, 'alphashapes.pickle'), 'rb') as f:
            alphashapes = pickle.load(f)
    else:
        alphashapes = None
    
    # Initialize the result using the 'reduce_initialization' function
    result = reduce_initialization(cellIDs, data_attrs, index, alphashapes, config)
    
    # Iterate over the cellIDs and update the result using the 'reduce_update' function
    for cellID in cellIDs:
        
        # Get the filename for the temporary chromosomal volumes of the cell
        filename = os.path.join(tempdir, '{}_result.pickle'.format(cellID))
        assert os.path.isfile(filename), "Parallel result file for cell {} not found.".format(cellID)
        
        # Load the cell file
        with open(filename, 'rb') as f:
            cell_result = pickle.load(f)
        
        # Update the result
        result = reduce_update(cellID, result, cell_result, cellIDs, data_attrs, index, alphashapes, config)
        
        del cell_result
    
    del data_attrs, index, alphashapes
    
    return result
