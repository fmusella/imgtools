import os
import sys
import pickle
import tempfile
from functools import partial
import typing
from alabtools.parallel import Controller
from .cte import ChromatinTracingExperiment
from .scf import SingleCellFeature
from . import utils

# TODO: Replace everywhere the usage of cte_parallel with this module

# Set the list of accepted modes
# Each mode corresponds to a different way of parallelizing the task.
#   - 'cell': each node performs the task on a single cell.
#   - 'chrom_pair': each node performs the task on a pair of chromosomes (across cells).
#   - 'triad': each node performs the task on a triad of [cell, chrom, trace].
ACCEPTED_MODES = ['cell', 'chrom_pair', 'triad']


# AUXILIARY FUNCTIONS

def check_config(config: dict, required_keys: dict, parallel: bool = True) -> None:
    """ Generic function for checking the config file for the parallelization tasks.

    Args:
        config (dict): config file for the parallelization tasks.
        required_keys (dict): dictionary of required keys for the config file.
        parallel (bool, optional): whether the parallelization is performed. Defaults to True.
    """
    
    if not isinstance(config, dict):
        raise TypeError(f'config should be a dictionary. Got type: {type(config)}')
    
    if not isinstance(required_keys, dict):
        raise TypeError(f'required_keys should be a dictionary. Got type: {type(required_keys)}')
    
    # Add the parallel key if parallel is True
    if parallel:
        required_keys['parallel'] = {'type': dict}
    
    for key in required_keys:
        # Check if the key is in the config
        if not key in config:
            raise ValueError(f'Key {key} not found in config.')
        # Check if the type of the key is correct (might be a list of types)
        if isinstance(required_keys[key]['type'], list):
            type_check = False
            for type in required_keys[key]['type']:
                if isinstance(config[key], type):
                    type_check = True
                    break
            if not type_check:
                raise TypeError(f"Invalid type for key: {key}. Got type: {type(config[key])}. Expected one of types: {required_keys[key]['type']}")
        else:
            if not isinstance(config[key], required_keys[key]['type']):
                raise TypeError(f"Invalid type for key: {key}. Got type: {type(config[key])}. Expected type: {required_keys[key]['type']}")
        # Check if numeric keys are positive
        if 'positive' in required_keys[key]:
            if not config[key] >= 0:
                raise ValueError(f'Key {key} should be positive. Got: {config[key]}')

def get_parallel_arguments(cte: ChromatinTracingExperiment, scf: SingleCellFeature, mode: str) -> list:
    """ Get the arguments for the parallelization task, depending on the mode.

    Args:
        cte (ChromatinTracingExperiment)
        scf (SingleCellFeature)
        mode (str): mode of the parallelization task.
            Accepted values: 'cell', 'chrom_pair', 'triad'.
            - 'cell': each argument is the cell ID.
            - 'chrom_pair': each argument is a tuple of (chrom_1, chrom_2).
            - 'triad': each argument is a tuple of (cell, chrom, trace).

    Returns:
        list: list of arguments for the parallelization task.
    """
    
    # Get the arguments for the parallelization task, depending on the mode
    # 1) 'cell': each argument is the cell ID
    if mode == 'cell':
        parallelIDs = list(cte.cell_labels) if cte is not None else list(scf.cell_labels)
    # 2) 'chrom_pair': each argument is a tuple of (chrom_1, chrom_2)
    elif mode == 'chrom_pair':
        # Get names of the chromosomes
        chroms = cte.index.genome.chroms if cte is not None else scf.index.genome.chroms
        # Get a list with all pairs of chromosomes
        parallelIDs = []
        for i in range(len(chroms)):
            for j in range(i, len(chroms)):
                parallelIDs.append((chroms[i], chroms[j]))
    # 3) 'triad': each argument is a tuple of (cell, chrom, trace)
    elif mode == 'triad':
        parallelIDs = cte.get_triad_labels() if cte is not None else scf.get_triad_labels()  # TODO: SCF doesn't have get_triad_labels yet!!
    # If the mode is not valid, raise an error
    else:
        raise ValueError(f'Invalid mode: {mode}. Accepted modes are: cell, chrom_pair, triad')
    
    return parallelIDs

def get_node_filename(parallelID: object, tempdir: str, mode: str) -> str:
    """ Get the filename for the node result based on the parallelID and the mode.

    Args:
        parallelID (object): either a cell ID (str), a tuple of chromosome names (tuple),
            or a triad (numpy.ndarray of shape (3,)).
        tempdir (str): temporary directory where the node results are saved.
        mode (str): mode of the parallelization task.

    Returns:
        str: filename for the node result.
    """
    
    # 1) mode = 'cell': parallelID is a cell ID (str)
    if mode == 'cell':
        cellID = parallelID  # cellID is a string
        filename = f'{cellID}.pickle'
    # 2) mode = 'chrom_pair': parallelID is a tuple of (chrom_1, chrom_2)
    elif mode == 'chrom_pair':
        chrom_1, chrom_2 = parallelID  # unpack the tuple
        filename = f'{chrom_1}_{chrom_2}.pickle'
    # 3) mode = 'triad': parallelID is a numpy.ndarray of shape (3,)
    elif mode == 'triad':
        cellID, chrom, traceID = parallelID  # unpack the numpy.ndarray
        filename = f'{cellID}_{chrom}_{traceID}.pickle'
    
    return os.path.join(tempdir, filename)


# MAIN FUNCTIONS

def control_func(
    cte: ChromatinTracingExperiment,
    scf: SingleCellFeature,
    config: dict,
    required_keys: dict,
    func_node: typing.Callable,
    reduce_initialization: typing.Callable,
    reduce_update: typing.Callable,
    mode : str = 'cell'
) -> object:
    """Generic function for controlling a parallelization task
    on a ChromatinTracingExperiment / SingleCellFeature object.
    
    Either CTE or SCF can be None, but not both.
    
    The parallelization can be performed in three modes:
        - across cells (mode='cell'): each node performs the task on a single cell.
        - across pairs of chromosomes (mode='chrom_pair'): each node performs the task on a pair of chromosomes (across cells).
        - across triads (mode='triad'): each node performs the task on a triad of [cell, chrom, trace].

    Args:
        cte (ChromatinTracingExperiment)
        scf (SingleCellFeature)
        config (dict): config file for the parallelization tasks.
        required_keys (dict): required keys for the config file.
        func_node (typing.Callable): cell task to perform on the node.
        reduce_initialization (typing.Callable): function to initialize the result object in the reduce task.
        reduce_update (typing.Callable): function to update the result object - with the cell results - in the reduce task.
        mode: (str, optional): mode of the parallelization task.
            Accepted values: 'cell', 'chrom_pair', 'triad'.
            - 'cell': each node performs the task on a single cell.
            - 'chrom_pair': each node performs the task on a pair of chromosomes (across cells).
            - 'triad': each node performs the task on a triad of [cell, chrom, trace].

    Returns:
        result (object): result of the parallelization task. Can be any object.
    """
    
    # Check that at least one between cte and scf is not None
    if cte is None and scf is None:
        raise ValueError('At least one between cte and scf should not be None.')
    
    # Check that the required keys are in the config
    check_config(config, required_keys)
    
    # Convert the paths in config to absolute paths
    utils.convert_to_abs_path(config)
    
    # Check that the mode is valid
    if mode not in ACCEPTED_MODES:
        raise ValueError(f'Invalid mode: {mode}. Accepted modes are: {ACCEPTED_MODES}')
    
    # Create a temporary directory
    tempdir = tempfile.mkdtemp(dir=os.getcwd())
    sys.stdout.write(f'Temporary directory for nodes results: {tempdir}\n')
    
    # create a Controller
    controller = Controller(config)
    
    # Get the names of CTE and SCF if they are not None
    cte_name = cte.h5_name if cte is not None else None
    scf_name = scf.h5_name if scf is not None else None
    
    # Get the arguments for the parallelization task, depending on the mode
    parallelIDs = get_parallel_arguments(cte, scf, mode)

    # run the parallel and reduce tasks
    parallel_task = partial(
        parallel_general,
        cte_name = cte_name,
        scf_name = scf_name,
        config=config,
        tempdir=tempdir,
        func_node=func_node,
        mode=mode
    )
    reduce_task = partial(
        reduce_general,
        cte_name = cte_name,
        scf_name = scf_name,
        config=config,
        tempdir=tempdir,
        reduce_initialization=reduce_initialization,
        reduce_update=reduce_update,
        mode=mode
    )
    result = controller.map_reduce(
        parallel_task,
        reduce_task,
        args = parallelIDs,
    )
    
    # Delete the non-empty temporary directory
    os.system(f'rm -r {tempdir}')
    
    del controller
    
    return result

def parallel_general(
    parallelID: object,
    cte_name: str,
    scf_name: str,
    config: dict,
    tempdir: str,
    func_node: typing.Callable,
    mode: str
) -> str:
    """ Generic function for performing a parallelization task on a single parallelization unit.
    
    The task is performed either on a cell, a pair of chromosomes, or a triad, depending on the mode.

    Args:
        parallelID (object): either a cell ID (str), a tuple of chromosome names (tuple),
            or a triad (numpy.ndarray of shape (3,)).
        cte_name (str)
        scf_name (str)
        config (dict)
        tempdir (str)
        func_node (typing.Callable)
        mode (str, optional): mode of the parallelization task.

    Returns:
        parallelID (object): the same parallelID that was passed as an argument.
    """
    
    # Perform the cell task on the node with the 'func_node' function
    node_result = func_node(parallelID, cte_name, scf_name, config)
    
    # Set the filename for the node result
    filename = get_node_filename(parallelID, tempdir, mode)
    
    # Save the node results in the temporary directory as a pickle file
    with open(filename, 'wb') as f:
        pickle.dump(node_result, f)
    
    del node_result
    
    return parallelID

def reduce_general(
    parallelIDs: list,
    cte_name: str,
    scf_name: str,
    config: dict,
    tempdir: str,
    reduce_initialization: typing.Callable,
    reduce_update: typing.Callable,
    mode: str
) -> object:
    """ Generic function for reducing the results of the parallelization task.

    Args:
        parallelIDs (list)
        cte_name (str)
        scf_name (str)
        config (dict)
        tempdir (str)
        reduce_initialization (typing.Callable)
        reduce_update (typing.Callable)

    Returns:
        result (object)
    """
    
    # Make sure that parallelIDs is a list and not empty
    assert isinstance(parallelIDs, list), f'parallelIDs should be a list. Got type: {type(parallelIDs)}'
    assert len(parallelIDs) > 0, 'parallelIDs should not be empty.'
    
    # Initialize the result using the 'reduce_initialization' function
    result = reduce_initialization(parallelIDs, cte_name, scf_name, config)
    
    # Iterate over the parallelIDs and update the result using the 'reduce_update' function
    for parallelID in parallelIDs:
        
        # Get the filename for the parallel result
        filename = get_node_filename(parallelID, tempdir, mode)
        assert os.path.exists(filename), f'Parallel result file not found: {filename}.'
        
        # Load the node result
        with open(filename, 'rb') as f:
            node_result = pickle.load(f)
        
        # Update the result
        result = reduce_update(parallelID, result, node_result, cte_name, scf_name, config)
        
        del node_result
    
    return result
