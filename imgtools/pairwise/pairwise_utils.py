import numpy as np
from alabtools.utils import Index
from ..cte import ChromatinTracingExperiment

def read_target_index(cte: ChromatinTracingExperiment, config: dict) -> Index:
    """ Read the target index from the configuration dictionary.
    
    The index is read based on the 'resolution' key in the config dictionary:
        - If 'resolution' is 'self', the target index is the CTE's index.
        - If 'resolution' is an integer, the target index is the CTE's index
          coarse-grained to that resolution.
        - If 'resolution' is a path to a HDF5 file, the target index is read from that file.
    
    If 'resolution' is not specified, it defaults to 'self'.

    Args:
        cte (ChromatinTracingExperiment)
        config (dict)

    Returns:
        Index: the target index of the output contact matrix.
    """
    
    # If there is no 'resolution' in the config, add the key 'self'
    if 'resolution' not in config:
        config['resolution'] = 'self'
    
    # If the 'resolution' is 'self', use the CTE's index
    if config['resolution'] == 'self':
        return cte.index
    
    # If the 'resolution' is a number, coarse-grain the CTE index to that resolution
    elif isinstance(config['resolution'], int):
        return cte.index.coarsegrain(config['resolution'])
    
    # If the 'resolution' is a path to a HDF5 file, read the index from that file
    elif isinstance(config['resolution'], str):
        try:
            return Index(config['resolution'])
        except Exception as e:
            raise ValueError(f"Could not read index from {config['resolution']}: {e}")
    
    # If the 'resolution' is not recognized, raise an error
    else:
        raise ValueError(
            f"Unrecognized resolution: {config['resolution']}.
            It should be 'self', an integer, or a path to an HDF5 file."
        )

def get_bins(chrom: str, starts: np.ndarray, ends: np.ndarray, domains_map: dict, index_map: dict) -> np.ndarray:
    """ Map the start/end coordinates first to the new start/end coordinates
    and then to the Index bins.
    
    We assume the chromosome is the same for all spots.

    Args:
        chrom (str): chromosome of all spots
        starts (np.ndarray): start coordinates of the spots
        ends (np.ndarray): end coordinates of the spots
        domains_map (dict): map of the start/end coordinates to the new start/end coordinates
        index_map (dict): map of the start/end coordinates to the Index bins

    Returns:
        bins (np.ndarray): bins of the spots in the Index
    """
    
    bins = []
    for start, end in zip(starts, ends):
        # Map start/end to new start/end
        _, start, end = domains_map[(chrom, start, end)][0]
        # Map the new start/end to the bins
        bin = index_map[(chrom, start, end)][0]
        bins.append(bin)
    bins = np.array(bins)
    
    return bins
