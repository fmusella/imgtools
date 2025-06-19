import os
import sys
import pickle
import tempfile
from functools import partial
import typing
from alabtools.parallel import Controller
from ..cte import ChromatinTracingExperiment
from .. import utils

def check_config(config: dict, required_keys: dict, parallel: bool = True) -> None:
    """ Generic function for checking the config file for the parallelization tasks.
    
    The required_keys dictionary has the following structure:
       { 'key1': {'type': type1, 'positive': True}, 'key2': ...}
    
    where type1 is either a type or a list of types,
    and 'positive' is an optional key that indicates that the value of the key should be positive.

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
            if not config[key] >= 0:
                raise ValueError("Key {} should be positive. Got: {}".format(key, config[key]))


def control_func(
    cte: ChromatinTracingExperiment,
    config: dict,
    required_keys: dict,
    func_node: typing.Callable,
    reduce_initialization: typing.Callable,
    reduce_update: typing.Callable
) -> object:
    
    # Check the required keys in the config
    check_config(config, required_keys, parallel=True)
    # Convert the paths in config to absolute paths
    utils.convert_to_abs_path(config)
    
    # Create a temporary directory
    tempdir = tempfile.mkdtemp(dir=os.getcwd())
    sys.stdout.write(f"Temporary directory for nodes' results: {tempdir}\n")
    
    # create a Controller
    controller = Controller(config)
    
    # Get names of the chromosomes
    chroms = cte.index.genome.chroms  # shape: (nchroms,)
    # Get a list with all pairs of chromosomes
    chrom_pairs = []
    for i in range(len(chroms)):
        for j in range(i, len(chroms)):
            chrom_pairs.append((chroms[i], chroms[j]))

    # run the parallel and reduce tasks
    parallel_task = partial(
        parallel_general,
        cte_name = cte.h5_name,
        config=config,
        tempdir=tempdir,
        func_node=func_node
    )
    reduce_task = partial(
        reduce_general,
        cte_name = cte.h5_name,
        config=config,
        tempdir=tempdir,
        reduce_initialization=reduce_initialization,
        reduce_update=reduce_update
    )
    result = controller.map_reduce(
        parallel_task,
        reduce_task,
        args = chrom_pairs,
    )
    
    # Delete the non-empty temporary directory
    os.system(f'rm -r {tempdir}')
    
    del controller
    
    return result

def parallel_general(
    chrom_pair: tuple, cte_name: str, config: dict, tempdir: str, func_node: typing.Callable
) -> tuple:
    
    # Get the chromosomes for the current pair
    chrom1, chrom2 = chrom_pair
    
    # Perform the task for the pair on the node with the 'func_node' function
    pair_result = func_node(chrom1, chrom2, cte_name, config, tempdir)
    
    # Save the pair results in the temporary directory as a pickle file
    out_filename = os.path.join(tempdir, f'{chrom1}_{chrom2}_result.pickle')
    with open(out_filename, 'wb') as f:
        pickle.dump(pair_result, f)
    
    del pair_result
    
    return chrom_pair

def reduce_general(
    chrom_pairs: list, cte_name: str, config: dict, tempdir: str,
    reduce_initialization: typing.Callable, reduce_update: typing.Callable
) -> object:
    
    assert isinstance(chrom_pairs, list), f"chrom_pairs should be a list. Got {type(chrom_pairs)}."
    assert len(chrom_pairs) > 0, "chrom_pairs should not be empty."
    
    # Initialize the result using the 'reduce_initialization' function
    result = reduce_initialization(chrom_pairs, cte_name, config)
    
    # Iterate over the chrom pairs and update the result using the 'reduce_update' function
    for chrom_pair in chrom_pairs:
        
        # Get the chromosomes for the current pair
        chrom1, chrom2 = chrom_pair
        
        # Get the filename for the current pair
        filename = os.path.join(tempdir, f'{chrom1}_{chrom2}_result.pickle')
        assert os.path.isfile(filename), f"Parallel result file for chromosome pair {chrom_pair} does not exist: {filename}"
        
        # Load the result for the current pair
        with open(filename, 'rb') as f:
            pair_result = pickle.load(f)
        
        # Update the result
        result = reduce_update(chrom1, chrom2, result, pair_result, cte_name, config, tempdir)
        
        del pair_result
    
    return result
