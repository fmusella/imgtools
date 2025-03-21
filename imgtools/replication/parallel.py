import os
import sys
import pickle
import tempfile
import typing
from functools import partial
import numpy as np
from alabtools.parallel import Controller
from ..scf import SingleCellFeature, scf_utils

def control_func(
    scf: SingleCellFeature, config: dict, func: typing.Callable,
    loci: np.array = None, p_c: np.array = None
) -> dict:
    """ Control function for the parallelization of a function (func) over the list of features in the SCF file.

    Args:
        scf (SingleCellFeature)
        config (dict): config file for the parallelization tasks.
        func_node (typing.Callable): function to be parallelized.

    Returns:
        dict: dictionary of results for each feature.
    """
    
    # Create a temporary directory
    tempdir = tempfile.mkdtemp(dir=os.getcwd())
    sys.stdout.write(f"Temporary directory for nodes' results: {tempdir}\n")
    
    # create a Controller
    controller = Controller(config)
    
    # Get the name of the SCF
    scf_name = scf.h5_name
    
    # If loci or p_c are provided, save them in the temporary directory
    for arr_str, arr in zip(['loci', 'p_c'], [loci, p_c]):
        if arr is not None:
            np.save(os.path.join(tempdir, f'{arr_str}.npy'), arr)

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
    
    # Try to read loci / p_c from the temporary directory, otherwise set it to None
    arrs = {'loci': None, 'p_c': None}
    for arr_str in arrs.keys():
        try:
            arrs[arr_str] = np.load(os.path.join(tempdir, f'{arr_str}.npy'))
        except FileNotFoundError:
            continue
    
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
    feat_result = func(N, Fq, states, config, arrs['loci'], arrs['p_c'])
    
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
