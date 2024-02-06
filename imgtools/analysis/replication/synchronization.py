import os
import sys
import time
from functools import partial
import numpy as np
import tempfile
import h5py
from scipy.stats import pearsonr
from alabtools.utils import Genome, Index
from alabtools.parallel import Controller
from ...scf import SingleCellFeature
from ... import utils


def run_synchronization(scf: SingleCellFeature, config: dict) -> (float, np.array):
    """ Synchronize the cell cycle with the replication timing in parallel.
    
    This method assumes that cells with lowest volume (bottom X%) are in G1,
    and cells with highest volume (top Y%) are in G2. X and Y have to be imputed.
    
    The imputation is done by optimizing the correlation coefficient between an external
    Replication Timing (RT) dataset and the RT computed from the SingleCellFeature object.
    
    The correlation during optimization is calculated on a subset of chromosomes (usechr in config),
    e.g. only odd chromosomes, so as to avoid overfitting.

    Args:
        scf (SingleCellFeature)
        config (dict): configuration dictionary.

    Returns:
        r (float): best optimization correlation coefficient between the RT and the cell cycle phase on the subset of chromosomes.
        cycle (np.array(ncell), dtype=str): best cell cycle array.
    """
    
    # Check that config is a dictionary
    assert isinstance(config, dict), "The input configuration must be a dictionary."
    
    # Check that the required keys are present in config
    required_keys = ['parallel', 'rt_bedfile', 'assembly', 'usechr', 'smooth', 'G1_n0', 'G1_n1', 'G2_n0', 'G2_n1']
    for key in required_keys:
        assert key in config.keys(), "The input configuration must have the key '{}'.".format(key)
    
    # create a temporary directory to store nodes' results
    temp_dir = tempfile.mkdtemp(dir=os.getcwd())
    sys.stdout.write("Temporary directory for nodes' results: {}\n".format(temp_dir))
    
    # create a Controller
    controller = Controller(config)
    
    # Read the RT data and assert that Index matches
    rt_bedfile = config['rt_bedfile']
    assembly = config['assembly']
    idx_rt = Index(rt_bedfile, genome=Genome(assembly))
    if not utils.compare_index(scf.index, idx_rt, config['usechr']):
        raise ValueError("The Index objects of the SingleCellFeature and the RT data do not match.")
    
    # compute all the possible G1/G2 segmentations and get the total number of segmentations
    segmentation = get_segmentation(config['G1_n0'], config['G1_n1'], config['G2_n0'], config['G2_n1'])
    nsegment = segmentation.shape[0]

    # set the parallel and reduce tasks
    parallel_task = partial(parallel_function,
                            scf_name=scf.h5_name,
                            cfg=config,
                            temp_dir=temp_dir)
    reduce_task = reduce_function

    # run the parallel and reduce tasks
    r, cycle = controller.map_reduce(parallel_task,
                                     reduce_task,
                                     args=np.arange(nsegment))
    
    # Delete the temporary directory and its contents
    os.system('rm -r {}'.format(temp_dir))
    
    return r, cycle


def parallel_function(segmentID: int, scf_name: str, cfg: dict, temp_dir: os.path) -> (float, str):
    """Parallel function for cell cycle imputation.
    
    It computes the Pearson correlation coefficient between the
    simulated RT signal and the experimental one for a given
    G1/S/G2 segmentation.
    
    Saves the cell-cycle state array and the Pearson correlation.
    
    The data is saved as a compressed numpy array.

    Args:
        segmentID (int): Segmentation ID (index of the segmentation
                                          in the segmentation array)
        cfg (dict): Configuration dictionary.
        temp_dir (str): Temporary directory where the data is stored.

    Returns:
        r (float): Pearson correlation coefficient for the given segmentation.
        out_name (str): Name of the output file.
    """
    
    # Get all possible G1/G2 segmentations
    segmentation = get_segmentation(cfg['G1_n0'], cfg['G1_n1'], cfg['G2_n0'], cfg['G2_n1'])
    
    # Read the SingleCellFeature object
    with h5py.File(scf_name, 'r') as f:
        index = Index(f)
        chromstr = index.chromstr
        volume = f['volumes'][:]
        ncount = f['spot_count'][:]
    
    # Number of cells in G1 and G2
    ncell_g1, ncell_g2 = segmentation[segmentID]
    ncell_g1, ncell_g2 = int(ncell_g1), int(ncell_g2)
    ncell = ncount.shape[0]
    
    # Define the cell cycle array (G1, S, G2)
    cycle = np.full(ncell, 'S', dtype='U10')
    cycle[:ncell_g1] = 'G1'
    cycle[(ncell - ncell_g2):] = 'G2'
    # The cell cycle array is sorted by volume (low to high)
    # Sort the cell cycle array back to the original order
    cycle = cycle[np.argsort(np.argsort(volume))]
    
    # Normalize the spots matrix (rho matrix)
    rho = normalize_bias(ncount, cycle)
    
    # Isolate the S phase submatrix
    rho_s = rho[cycle == 'S', :, :]
    
    # Compute the simulated RT signal
    rt_sim = np.nansum(rho_s, axis=(0, 2))
    
    # Smooth the simulated RT signal if specified in cfg
    if cfg['smooth']:
        try:
            k = cfg['k']
        except:
            raise ValueError("k must be specified in cfg if smooth is True")
        rt_sim = utils.smooth(rt_sim, chromstr, k)
    
    # Read the experimental RT signal
    rt_bedfile = cfg['rt_bedfile']
    assembly = cfg['assembly']
    idx_exp = Index(rt_bedfile, genome=Genome(assembly))
    try:
        rt_exp = idx_exp.track0
    except:
        raise ValueError("{} must be a BedGraph, \
            with a single track and no header".format(rt_bedfile))
    
    # Isolate the RT signals for chromosomes specified in cfg['usechr']
    usechr = cfg['usechr']  # should contain only even or odd autosomes
    rt_sim_usechr = rt_sim[np.isin(chromstr, usechr)]
    rt_exp_usechr = rt_exp[np.isin(idx_exp.chromstr, usechr)]
    
    # Compute the Pearson correlation coefficient
    r = utils.clean_pearsonr(rt_sim_usechr, rt_exp_usechr)
    
    # Save the cycle as a compressed numpy array
    out_name = os.path.join(temp_dir, '{}.npz'.format(segmentID))
    np.savez_compressed(out_name, cycle=cycle)

    # Free memory
    del segmentation, ncount, volume, cycle, rho, rho_s, rt_sim, rt_exp, idx_exp, rt_sim_usechr, rt_exp_usechr
    
    return r, out_name

def reduce_function(parallel_returns: list) -> (float, np.array):
    """Reduce function for cell cycle imputation.
    
    Determines the best segmentation based on largest
    Pearson correlation coefficient from all possible segmentations.
    
    Returns the best cycle and Pearson correlation.

    Args:
        parallel_returns (list): List of the parallel returns.

    Returns:
        r_best (float): Pearson correlation coefficient.
        cycle_best (np.array(ncell), dtype=int): Cell cycle array.
    """
    
    sys.stdout.write('Starting reduce function\n')
    
    # Find the best segmentation
    best = {'r': 0, 'out_name': ''}
    for parallel_return in parallel_returns:
        r, out_name = parallel_return
        if r > best['r']:
            best['r'] = r
            best['out_name'] = out_name
    
    if best['r'] == 0:
        raise ValueError("No segmentation with r > 0 found")
    
    # Read the cell cycle array of the best segmentation
    cycle_best = np.load(best['out_name'])['cycle']
    
    return best['r'], cycle_best


# Auxiliary functions

def get_segmentation(min_g1: int, max_g1: int, min_g2: int, max_g2: int) -> np.array:
    """ Get all the possible G1/G2 segmentations.

    Args:
        min_g1 (int): minimum number of cells in G1.
        max_g1 (int): maximum number of cells in G1.
        min_g2 (int): minimum number of cells in G2.
        max_g2 (int): maximum number of cells in G2.

    Returns:
        segmentation (np.array(nsegment, 2), dtype=int): segmentation array.
                Each row is a possible G1/G2 segmentation, i.e. [ncell_g1, ncell_g2].
    """
    
    # Initialize the segmentation array
    segmentation = []
    
    # Loop over all the possible G1/G2 segmentations and append them to the segmentation array
    for ncell_g1 in range(min_g1, max_g1):
        for ncell_g2 in range(min_g2, max_g2):
            segmentation.append([ncell_g1, ncell_g2])
    segmentation = np.array(segmentation)
    
    return segmentation


def normalize_bias(ncount: np.array, cycle: np.array) -> np.array:
    """Normalize the bias in the raw spots counts.

    Args:
        ncount (np.array(ncell, ndomain, ncopy_max), dtype=int): raw single-cell spot counts.
        cycle (np.array(ndomain), dtype='U10'): cell cycle (G1, S, G2) array.

    Returns:
        rho (np.array(ncell, ndomain, ncopy_max), dtype=float): normalized single-cell spot counts.
                                                                It is a float signal.
    """
    
    # If cycle doesn't have G1 or G2 cells, throw an error
    if not np.any(cycle == 'G1') or not np.any(cycle == 'G2'):
        raise ValueError("cycle must have G1 and G2 cells")
    
    # Assert that the input arrays have the correct shape
    ncell, ndomain, _ = ncount.shape
    assert cycle.shape[0] == ncell,\
        "ncount and cycle must have the same number of cells"
    
    # Isolate G1 and G2 raw spots    
    ncount_g1 = ncount[cycle == 'G1', :, :]
    ncount_g2 = ncount[cycle == 'G2', :, :]
    
    # Compute the bias arrays
    # Since the cells in G1 and G2 are not replicating,
    # variation in the total number of spots is due noise or bias.
    # If we see that a domain has systematically more/less spots than others in G1 or G2,
    # we can assume that this is due to bias and not noise
    # (for example GC rich domains are detected more likely than AT rich domains).
    # Therefore, we can estimate the bias by computing the total number of spots
    # in each domain in G1 and G2.
    bias_g1 = np.nansum(ncount_g1, axis=(0, 2))  # np.array(ndomain)
    bias_g2 = np.nansum(ncount_g2, axis=(0, 2))
    
    # Rescale the bias arrays to have mean 1
    bias_g1 = bias_g1 / np.nanmean(bias_g1)
    bias_g2 = bias_g2 / np.nanmean(bias_g2)
    
    # Set the total bias as mean of the G1 and G2 biases
    bias = (bias_g1 + bias_g2) / 2
    
    # If bias_g1 has NaNs, set the bias as bias_g2 and vice versa
    bias[np.isnan(bias_g1)] = bias_g2[np.isnan(bias_g1)]
    bias[np.isnan(bias_g2)] = bias_g1[np.isnan(bias_g2)]
    
    # Rescale the bias to have mean 1
    # (again, since NaNs could have screwed up the mean)
    bias = bias / np.nanmean(bias)
    
    # Reshape the bias array to be able to broadcast it
    bias = np.reshape(bias, (1, ndomain, 1))  # np.array(1, ndomain, 1)
    
    # Compute the normalized spots matrix
    rho = np.copy(ncount)
    rho = rho / bias
    
    return rho
