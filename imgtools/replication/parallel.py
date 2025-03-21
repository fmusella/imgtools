import os
import sys
import pickle
import tempfile
import typing
from functools import partial
from alabtools.parallel import Controller
from ..scf import SingleCellFeature, scf_utils

def check_config(config: dict, required_keys: dict, parallel: bool = True) -> None:
    """ Check the configuration file for the parallelization tasks.

    Args:
        config (dict): config file for the parallelization tasks.
        required_keys (dict): dictionary of required keys for the config file.
        parallel (bool, optional): whether the parallelization is performed. Defaults to True.
    """
    
    # Check that config and required_keys are dictionaries
    if not isinstance(config, dict):
        raise TypeError(f"config should be a dictionary. Got type: {type(config)}")
    if not isinstance(required_keys, dict):
        raise TypeError(f"required_keys should be a dictionary. Got type: {type(required_keys)}")
    
    # Add the parallel key if parallel is True
    if parallel:
        required_keys['parallel'] = {'type': dict}
    
    # Loop over the required keys
    for key in required_keys:
        # Check if the key is in the config
        if not key in config:
            raise ValueError(f"Key {key} not found in config.")
        # Check if the type of the key is correct (might be a list of types)
        if isinstance(required_keys[key]['type'], list):
            type_check = False
            for type in required_keys[key]['type']:
                if isinstance(config[key], type):
                    type_check = True
                    break
            if not type_check:
                raise TypeError(f"Invalid type for key: {key}. Got type: {type(config[key])}. Expected type: {required_keys[key]['type']}")
        else:
            if not isinstance(config[key], required_keys[key]['type']):
                raise TypeError(f"Invalid type for key: {key}. Got type: {type(config[key])}. Expected type: {required_keys[key]['type']}")
        # Check if numeric keys are positive
        if 'positive' in required_keys[key]:
            if not config[key] > 0:
                raise ValueError(f"Key {key} should be positive. Got: {config[key]}")

def control_func(scf: SingleCellFeature, config: dict, required_keys: dict, func: typing.Callable) -> dict:
    """ Control function for the parallelization of a function (func) over the list of features in the SCF file.

    Args:
        scf (SingleCellFeature)
        config (dict): config file for the parallelization tasks.
        required_keys (dict): dictionary of required keys for the config file.
        func_node (typing.Callable): function to be parallelized.

    Returns:
        dict: dictionary of results for each feature.
    """
    
    # Check the configuration
    check_config(config, required_keys)
    
    # Create a temporary directory
    tempdir = tempfile.mkdtemp(dir=os.getcwd())
    sys.stdout.write(f"Temporary directory for nodes' results: {tempdir}\n")
    
    # create a Controller
    controller = Controller(config)
    
    # Get the name of the SCF
    scf_name = scf.h5_name

    # run the parallel and reduce tasks
    parallel_task = partial(
        parallel_func,
        scf_name = scf_name,
        config=config,
        tempdir=tempdir,
        func=func
    )
    reduce_task = partial(
        reduce_func,
        tempdir=tempdir,
    )
    result = controller.map_reduce(
        parallel_task,
        reduce_task,
        args = scf.feature_list
    )
    
    # Delete the non-empty temporary directory
    os.system(f'rm -r {tempdir}')
    
    return result

def parallel_func(feat: str, scf_name: str, config: dict, tempdir: str, func: typing.Callable) -> str:
    """ Node-level function for the parallelization of a function (func) on a feature in the SCF file.

    Args:
        feat (str)
        scf_name (str): name of the SCF file.
        config (dict): config file for the parallelization tasks.
        tempdir (str): temporary directory for the node's results.
        func (typing.Callable): function to be parallelized.

    Returns:
        str: name of the feature.
    """

    # Load the SCF
    scf = SingleCellFeature(scf_name, 'r')
    # Get the index
    index = scf.index
    # Get the states
    states = scf.cell_states
    
    # Get the 'spotcount' and feature data
    N = scf.get_feature('spotcount')  # shape: (ncells, nloci, ncopies)
    F = scf.get_feature(feat)  # shape: (ncells, nloci, ncopies)
    
    # Curate missing chromosomes
    scf_utils.curate_missing_chromosomes(N, index)
    scf_utils.curate_missing_chromosomes(F, index)
    
    # Quantize the feature matrix
    nquants = config['nquants']
    Fq, _ = scf_utils.quantize_matrix(F, nquants)
    del F

    # Perform the feature run
    feat_result = func(N, Fq, states, config)
    
    # Save the feature result in the temporary directory as a pickle file
    out_filename = os.path.join(tempdir, f'{feat}_result.pickle')
    with open(out_filename, 'wb') as f:
        pickle.dump(feat_result, f)
    
    return feat

def reduce_func(features: list, tempdir: str) -> dict:
    """ Reduce function for the parallelization of a function on a list of features in the SCF file.

    Args:
        features (list)
        tempdir (str): temporary directory for the node's results.

    Returns:
        dict: dictionary of results for each feature.
    """
    
    assert isinstance(features, list), f"'features' should be a list. Got type: {type(features)}"
    assert len(features) > 0, "'features' should not be empty."
    
    # Initialize the results for all features
    result = {}
    
    # Iterate over the features and update the results
    for feat in features:
        
        # Get the filename for the cell
        filename = os.path.join(tempdir, f'{feat}_result.pickle')
        assert os.path.isfile(filename), f"Parallel result file for feature '{feat}' not found."
        
        # Load the feature result
        with open(filename, 'rb') as f:
            feat_result = pickle.load(f)
        
        # Update the result
        result[feat] = feat_result
    
    return result
