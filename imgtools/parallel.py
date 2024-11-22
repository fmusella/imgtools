import os
import sys
import pickle
import tempfile
from functools import partial
import typing
from alabtools.parallel import Controller
from .cte import ChromatinTracingExperiment
from .scf import SingleCellFeature

# TODO: 1) Add the option to parallelize of traces or chromosomes, e.g. triadIDs?
#       2) Replace everywhere the usage of cte_parallel with this module


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
    
    # Add the parallel key if parallel is True
    if parallel:
        required_keys['parallel'] = {'type': dict}
    
    for key in required_keys:
        # Check if the key is in the config
        if not key in config:
            raise ValueError("Key {} not found in config.".format(key))
        # Check if the type of the key is correct (might be a list of types)
        if isinstance(required_keys[key]['type'], list):
            type_check = False
            for type in required_keys[key]['type']:
                if isinstance(config[key], type):
                    type_check = True
                    break
            if not type_check:
                raise TypeError("Invalid type for key: {}. Got type: {}. Expected type: {}".format(key, type(config[key]), required_keys[key]['type']))
        else:
            if not isinstance(config[key], required_keys[key]['type']):
                raise TypeError("Invalid type for key: {}. Got type: {}. Expected type: {}".format(key, type(config[key]), required_keys[key]['type']))
        # Check if numeric keys are positive
        if 'positive' in required_keys[key]:
            if not config[key] > 0:
                raise ValueError("Key {} should be positive. Got: {}".format(key, config[key]))


def control_func(
    cte: ChromatinTracingExperiment,
    scf: SingleCellFeature,
    config: dict,
    required_keys: dict,
    func_node: typing.Callable,
    reduce_initialization: typing.Callable,
    reduce_update: typing.Callable
) -> object:
    """Generic function for controlling a parallelization task
    on a ChromatinTracingExperiment / SingleCellFeature object.

    Args:
        cte (ChromatinTracingExperiment)
        scf (SingleCellFeature)
        config (dict): config file for the parallelization tasks.
        required_keys (dict): required keys for the config file.
        func_node (typing.Callable): cell task to perform on the node.
        reduce_initialization (typing.Callable): function to initialize the result object in the reduce task.
        reduce_update (typing.Callable): function to update the result object - with the cell results - in the reduce task.

    Returns:
        result (object): result of the parallelization task. Can be any object.
    """
    
    # Check that at least one between cte and scf is not None
    if cte is None and scf is None:
        raise ValueError("At least one between cte and scf should not be None.")
    
    # Check that the required keys are in the config
    check_config(config, required_keys)
    
    # Create a temporary directory
    tempdir = tempfile.mkdtemp(dir=os.getcwd())
    sys.stdout.write("Temporary directory for nodes' results: {}\n".format(tempdir))
    
    # create a Controller
    controller = Controller(config)
    
    # Get the names of CTE and SCF if they are not None
    cte_name = cte.h5_name if cte is not None else None
    scf_name = scf.h5_name if scf is not None else None

    # run the parallel and reduce tasks
    parallel_task = partial(
        parallel_general,
        cte_name = cte_name,
        scf_name = scf_name,
        config=config,
        tempdir=tempdir,
        func_node=func_node
    )
    reduce_task = partial(
        reduce_general,
        cte_name = cte_name,
        scf_name = scf_name,
        config=config,
        tempdir=tempdir,
        reduce_initialization=reduce_initialization,
        reduce_update=reduce_update
    )
    result = controller.map_reduce(
        parallel_task,
        reduce_task,
        args = list(cte.cell_labels)
    )
    
    # Delete the non-empty temporary directory
    os.system('rm -r {}'.format(tempdir))
    
    del controller
    
    return result

def parallel_general(
    cellID: str,
    cte_name: str,
    scf_name: str,
    config: dict,
    tempdir: str,
    func_node: typing.Callable
) -> str:
    """ Generic function for performing a parallelization task on a single cell.

    Args:
        cellID (str)
        cte_name (str)
        scf_name (str)
        config (dict)
        tempdir (str)
        func_node (typing.Callable)

    Returns:
        cellID (str)
    """
    
    # Perform the cell task on the node with the 'func_node' function
    cell_result = func_node(cellID, cte_name, scf_name, config)
    
    # Save the cell results in the temporary directory as a pickle file
    out_filename = os.path.join(tempdir, '{}_result.pickle'.format(cellID))
    with open(out_filename, 'wb') as f:
        pickle.dump(cell_result, f)
    
    del cell_result
    
    return cellID

def reduce_general(
    cellIDs: list,
    cte_name: str,
    scf_name: str,
    config: dict,
    tempdir: str,
    reduce_initialization: typing.Callable,
    reduce_update: typing.Callable
) -> object:
    """ Generic function for reducing the results of the parallelization task.

    Args:
        cellIDs (list)
        cte_name (str)
        scf_name (str)
        config (dict)
        tempdir (str)
        reduce_initialization (typing.Callable)
        reduce_update (typing.Callable)

    Returns:
        result (object)
    """
    
    assert isinstance(cellIDs, list), "cellIDs should be a list. Got type: {}".format(type(cellIDs))
    assert len(cellIDs) > 0, "cellIDs should not be empty."
    
    # Initialize the result using the 'reduce_initialization' function
    result = reduce_initialization(cellIDs, cte_name, scf_name, config)
    
    # Iterate over the cellIDs and update the result using the 'reduce_update' function
    for cellID in cellIDs:
        
        # Get the filename for the cell
        filename = os.path.join(tempdir, '{}_result.pickle'.format(cellID))
        assert os.path.isfile(filename), "Parallel result file for cell {} not found.".format(cellID)
        
        # Load the cell file
        with open(filename, 'rb') as f:
            cell_result = pickle.load(f)
        
        # Update the result
        result = reduce_update(cellID, result, cell_result, cte_name, scf_name, config)
        
        del cell_result
    
    return result
