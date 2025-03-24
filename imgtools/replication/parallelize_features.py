import os
import sys
import pickle
import tempfile
import h5py
from functools import partial
import numpy as np
from alabtools.parallel import Controller
from ..scf import SingleCellFeature, scf_utils
from ..utils import clip_array
from .GMM_solver import GMM_solve

def control_func(scf: SingleCellFeature, config: dict, arrays: dict = {}) -> dict:
    """ Control function for the parallelization of feature-dependent analyses
    over the features in a SCF file.

    Args:
        scf (SingleCellFeature)
        config (dict): config file for the parallelization tasks.
        arrays (dict): dictionary of arrays to save in the temporary directory. Default: {}.

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
    
    # If provided, save the additional arrays in the temporary directory in a h5 file
    if len(arrays) > 0:
        with h5py.File(os.path.join(tempdir, 'arrays.h5'), 'w') as f:
            for arr_str, arr in arrays.items():
                f.create_dataset(arr_str, data=arr)
    
    # run the parallel and reduce tasks
    parallel_task = partial(
        parallel_func,
        scf_name = scf_name,
        config=config,
        tempdir=tempdir
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

def parallel_func(feat: str, scf_name: str, config: dict, tempdir: str) -> str:
    """ Node-level function for the parallelization of a function (func) on a feature in the SCF file.

    Args:
        feat (str)
        scf_name (str): name of the SCF file.
        config (dict): config file for the parallelization tasks.
        tempdir (str): temporary directory for the node's results.

    Returns:
        str: name of the feature.
    """

    # Load the SCF
    scf = SingleCellFeature(scf_name, 'r')
    # Get the index
    index = scf.index
    # Get the states
    states = scf.cell_states
    
    # If 'arrays.h5' exists in the temporary directory, load it
    try:
        arrays_h5 = h5py.File(os.path.join(tempdir, 'arrays.h5'), 'r')
    except FileNotFoundError:
        arrays_h5 = None
    
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
    feat_result = single_feat_func(N, Fq, states, config, arrays_h5)
    
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

def single_feat_func(N: np.ndarray, Fq: np.ndarray, states: np.ndarray, config: dict, arrays_h5: h5py.File = None) -> dict:
    """ Node function to perform the feature-dependent analysis for a single feature.
    
    Calculates the replication probability for all loci in the same feature quantile.

    Args:
        N (np.ndarray): number of spots. shape: (ncells, nloci, ncopies)
        Fq (np.ndarray): quantized feature. shape: (ncells, nloci, ncopies)
        states (np.ndarray): cell states. shape: (ncells,)
        config (dict): configuration dictionary with the following keys:
            - nquants (int): number of quantiles for the feature.
            - S_stage (tuple, optional): tuple with the lower and upper bounds of the S stage. Default: None.
                - if None, all S cells are considered.
            - re-weighting (bool, optional): re-weight the efficiency and bias in S according to p_q_S. Default: False.
        arrays_h5 (h5py.File): h5py file with additional arrays. Default: None.

    Returns:
        dict: results of the feature-dependent analysis, with the following keys:
            - eps_q_G1, detection efficiency in G1. shape: (nquants),
            - eps_q_G1_err, error in eps_q_G1. shape: (nquants),
            - beta_q_G1, bias rate in G1. shape: (nquants),
            - beta_q_G1_err, error in beta_q_G1. shape: (nquants),
            - eps_q_G2, detection efficiency in G2. shape: (nquants),
            - eps_q_G2_err, error in eps_q_G2. shape: (nquants),
            - beta_q_G2, bias rate in G2. shape: (nquants),
            - beta_q_G2_err, error in beta_q_G2. shape: (nquants),
            - eps_q_S, detection efficiency in S. shape: (nquants),
            - eps_q_S_err, error in eps_q_S. shape: (nquants),
            - beta_q_S, bias rate in S. shape: (nquants),
            - beta_q_S_err, error in beta_q_S. shape: (nquants),
            - p_q_S, replication probability in S. shape: (nquants),
            - p_q_S_err, error in p_q_S. shape: (nquants).
    """
    
    # Get the parameters from the config dictionary
    nquants = config['nquants']
    
    # If 'p_c' is present in arrays_h5, load it
    if arrays_h5 is not None and 'p_c' in arrays_h5:
        p_c = arrays_h5['p_c'][:]
    # If 'loci' is present in arrays_h5, load it
    if arrays_h5 is not None and 'loci' in arrays_h5:
        loci = arrays_h5['loci'][:]
    
    # Initialize the summary statistics dictionary
    stat = {}
    
    # Loop over the states
    for s in ['G1', 'S', 'G2']:
        
        # Get the mask for the state
        mask_state = states == s
        
        # If the state is S AND the S_stage is provided, filter the S cells in the S_stage
        if s == 'S' and 'S_stage' in config:
            S_stage = config['S_stage']
            mask_state = np.logical_and(mask_state, np.logical_and(p_c > S_stage[0], p_c < S_stage[1]))
        
        # TODO: mask for volumes < 400 um^3?
        
        # Mask for the state
        N_s = N[mask_state, :, :]
        Fq_s = Fq[mask_state, :, :]
        
        # If loci are provided, filter the loci
        if loci is not None:
            N_s = N_s[:, loci, :]
            Fq_s = Fq_s[:, loci, :]
        
        # TODO: remove z and envdist quantiles?
        
        # Initialize the arrays to store quantile-dependent averages
        stat[s] = {
            'nsamples': np.zeros(nquants),  # shape: (nquants)
            'n': np.zeros(nquants),
            'n_var': np.zeros(nquants),
            'f': np.zeros(nquants),
            'f_var': np.zeros(nquants),
            'nf_cov': np.zeros(nquants)
        }
        
        # Loop over the quantiles
        for q in range(nquants):
            
            # Mask for the quantile
            mask_q = Fq_s == q
            N_s_q = N_s[mask_q]
            
            # Create a zero-indicator version of N_s_q: 1 if N_s = 0, 0 otherwise
            B_s_q = (N_s_q == 0).astype(float)
            B_s_q[np.isnan(N_s_q)] = np.nan
            
            # Calculate average/std for the quantile
            nsamples = np.sum(~np.isnan(N_s_q))  # int
            stat[s]['nsamples'][q] = nsamples
            stat[s]['n'][q] = np.nanmean(N_s_q)
            stat[s]['n_var'][q] = np.nanvar(N_s_q, ddof=1) / nsamples
            stat[s]['f'][q] = np.nanmean(B_s_q)
            stat[s]['f_var'][q] = np.nanvar(B_s_q, ddof=1) / nsamples
            stat[s]['nf_cov'][q] = - stat[s]['n'][q] * stat[s]['f'][q] / nsamples
    
    # Calculate efficiency and bias in G1 and G2
    eps_q_G1, beta_q_G1, eps_q_G1_err, beta_q_G1_err = GMM_solve(stat['G1'], p='G1')
    eps_q_G2, beta_q_G2, eps_q_G2_err, beta_q_G2_err = GMM_solve(stat['G2'], p='G2')
    eps_q_G1 = clip_array(eps_q_G1, 0, 1)
    eps_q_G2 = clip_array(eps_q_G2, 0, 1)
    beta_q_G1 = clip_array(beta_q_G1, 0, None)
    beta_q_G2 = clip_array(beta_q_G2, 0, None)
    
    # We assume that the efficiency in S is the average of G1 and G2
    eps_q_S = (eps_q_G1 + eps_q_G2) / 2
    eps_q_S_err = np.sqrt(eps_q_G1_err ** 2 + eps_q_G2_err ** 2) / 2
    
    # Calculate replication probability and bias in S
    p_q_S, beta_q_S, p_q_S_err, beta_q_S_err = GMM_solve(stat['S'], eps=eps_q_S, eps_err=eps_q_S_err)
    p_q_S = clip_array(p_q_S, 0, 1)
    beta_q_S = clip_array(beta_q_S, 0, None)
    
    # If 're-weighting' in config is True, re-weight the efficiency and bias in S according to p_q_S
    if 're-weighting' in config and config['re-weighting']:
        
        # We re-calculate the efficiency and bias in S
        eps_q_S = (eps_q_G1 * (1 - p_q_S) + eps_q_G2 * p_q_S)
        beta_q_S = (beta_q_G1 * (1 - p_q_S) + beta_q_G2 * p_q_S)
        eps_q_S_err = np.sqrt(eps_q_G1_err**2 * (1 - p_q_S)**2 + eps_q_G2_err**2 * p_q_S**2)
        beta_q_S_err = np.sqrt(beta_q_G1_err**2 * (1 - p_q_S)**2 + beta_q_G2_err**2 * p_q_S**2)
        
        # And we re-calculate the replication probability using the new efficiency
        p_q_S, beta_q_S, p_q_S_err, beta_q_S_err = GMM_solve(stat['S'], eps=eps_q_S, eps_err=eps_q_S_err)
        p_q_S = clip_array(p_q_S, 0, 1)
        beta_q_S = clip_array(beta_q_S, 0, None)
    
    # Return the results as a dictionary
    return {
        'eps_q_G1': eps_q_G1,
        'eps_q_G1_err': eps_q_G1_err,
        'beta_q_G1': beta_q_G1,
        'beta_q_G1_err': beta_q_G1_err,
        'eps_q_G2': eps_q_G2,
        'eps_q_G2_err': eps_q_G2_err,
        'beta_q_G2': beta_q_G2,
        'beta_q_G2_err': beta_q_G2_err,
        'eps_q_S': eps_q_S,
        'eps_q_S_err': eps_q_S_err,
        'beta_q_S': beta_q_S,
        'beta_q_S_err': beta_q_S_err,
        'p_q_S': p_q_S,
        'p_q_S_err': p_q_S_err
    }
