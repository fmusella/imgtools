import os
import pickle
import numpy as np
from . import utils
from scipy.spatial.distance import cdist


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


# PARALLEL FUNCTIONS FOR THE HOMOLOGUES PROXIMITY TASK

required_keys_homoprox = {
    'proximity_threshold': {'type': float, 'positive': True},
}

def do_chrom_homoprox(chrom_data: dict, proximity_threshold: float):
    """ Checks if the homologues of a chromosome are close to each other.

    Args:
        chrom_data (dict): The data of a single chromosome in dictionary format.
        proximity_threshold (float): The maximum distance between two homologues for them to be considered proximal.

    Returns:
        bool: True if the homologues are close, False otherwise.
    """
    
    for i1, traceID_1 in enumerate(chrom_data):
        for i2, traceID_2 in enumerate(chrom_data):
            
            # Avoid comparing the same pair of traces twice (and avoid comparing a trace to itself)
            if i1 >= i2:
                continue
            
            # Convert the data to numpy arrays
            xs1, ys1, zs1, _, _, _, _, _ = utils.trace_dict_to_numpy(chrom_data[traceID_1])
            xs2, ys2, zs2, _, _, _, _, _ = utils.trace_dict_to_numpy(chrom_data[traceID_2])
            
            # Calculate the minimum distance between the two traces
            crd1 = np.array([xs1, ys1, zs1]).T
            crd2 = np.array([xs2, ys2, zs2]).T
            min_dist = np.min(cdist(crd1, crd2))
            
            # If the minimum distance is below the threshold, we have found a pair of proximal homologues
            if min_dist <= proximity_threshold:
                return True
    
    # If we get here, no proximal homologues were found
    return False
            
def homoprox_parallel(cellID: str, config: dict, tempdir: str):
    """ Parallel function for the homologues proximity task.
    For each chromosome, checks if the homologues are close to each other.
    Saves the results (dictionary) as a pickle file in the tempdir.

    Args:
        cellID (str)
        config (dict): config file for the homologues proximity task.
        tempdir (str): temporary directory for storing intermediate results.

    Returns:
        cellID (str)
    """

    check_config(config, required_keys_homoprox)
    
    assert isinstance(cellID, str), "cellID should be a string. Got type: {}".format(type(cellID))
    
    assert isinstance(tempdir, str), "tempdir should be a string. Got type: {}".format(type(tempdir))
    assert os.path.isdir(tempdir), "tempdir should be a directory. Got: {}".format(tempdir)
    
    # Try to load the data for the cell with pickle
    in_filename = os.path.join(tempdir, '{}_data.pickle'.format(cellID))
    assert os.path.isfile(in_filename), "Data for cell {} not found.".format(cellID)
    with open(in_filename, 'rb') as f:
        cell_data = pickle.load(f)
    
    # Initialize output
    prox_bool = {}  # for each chromosome, False if no homologues are close, True if they are
    
    # Perform tracing on each chromosome
    for chrom in cell_data:

        # Skip chromosomes with less than 2 traces
        if len(cell_data[chrom]) < 2:
            continue
        
        # If the homologues of chrom are close, prox_bool[chrom] will be True, otherwise False
        prox_bool[chrom] = do_chrom_homoprox(cell_data[chrom], config['proximity_threshold'])
    
    # Save prox_bool as a pickle file
    out_filename = os.path.join(tempdir, '{}_prox_bool.pickle'.format(cellID))
    with open(out_filename, 'wb') as f:
        pickle.dump(prox_bool, f)
    
    del cell_data, prox_bool
    
    return cellID

def homoprox_reduce(cellIDs: list, tempdir: str):
    
    assert isinstance(cellIDs, list), "cellIDs should be a list. Got type: {}".format(type(cellIDs))
    assert len(cellIDs) > 0, "cellIDs should not be empty."
    
    prox_count = {}
    total_count = {}

    for cellID in cellIDs:
        
        # Get the filename for the prox_bool of the cell
        filename = os.path.join(tempdir, '{}_prox_bool.pickle'.format(cellID))
        
        assert os.path.isfile(filename), "prox_bool file for cell {} not found.".format(cellID)

        with open(filename, 'rb') as f:
            cell_prox_bool = pickle.load(f)
        
        for chrom in cell_prox_bool:
            
            # Increment the total count for the current chromosome
            if chrom in total_count:
                total_count[chrom] += 1
            else:
                total_count[chrom] = 1
            
            # If the homologues are proximal, increment the proximal count for the current chromosome
            if cell_prox_bool[chrom]:
                if chrom in prox_count:
                    prox_count[chrom] += 1
                else:
                    prox_count[chrom] = 1
    
    # Calculate the homologue proximity ratio for each chromosome
    ratio = {}
    
    for chrom in total_count:
        
        # If the chromosome has never been found to have proximal homologues, the ratio is 0
        if chrom not in prox_count:
            ratio[chrom] = 0
            continue
        
        # Otherwise, compute the ratio
        ratio[chrom] = prox_count[chrom] / total_count[chrom]
    
    return ratio
