# STILL USING CTE_PARALLEL INSTEAD OF PARALLEL.
# I HAVE TO REPLACE IT WITH PARALLEL IN THE FUTURE.

# ALSO, I HAVE TO DECIDE IF I WANT TO KEEP THIS MODULE HERE.
# I CREATED THE FOLDER 'PROCESSING'. THE REASON IS THAT I DON'T LIKE
# THAT THIS MODULE CALLS BOTH CTE AND PARALLEL - WHO ITSELF CALLS CTE,
# AND YET THIS MODULE IS INSIDE THE CTE FOLDER.
# SINCE THIS FUNCTION CALLS SOMETHING OUTSIDE THE CTE FOLDER, WHICH IS NOT UTILS,
# I THINK IT'S MORE APPROPRIATE TO MOVE IT OUTSIDE CTE.


import numpy as np
import h5py
from copy import deepcopy
from collections import defaultdict
from scipy.spatial.distance import cdist
from alabtools.utils import map_indices
from . import cte_io
from .cte import ChromatinTracingExperiment
from . import cte_utils
from . import cte_parallel
from . import metrics
from ..tracing import GenomicIterativeDBSCAN
from ..tracing import WardSpectralClustering
from .. import utils


# TRACING

tracing_available_methods = ['gidbscan', 'wsclustering']

tracing_required_keys = {
    'gidbscan': {
        'cte_traced_name': {'type': str},
        'dbscan_eps': {'type': float, 'positive': True},
        'dbscan_min_samples': {'type': int, 'positive': True},
        'window_size': {'type': int, 'positive': True},
        'delta': {'type': float, 'positive': True},
        'merging_proximity_length': {'type': int, 'positive': True},
        'merging_overlap_threshold': {'type': float, 'positive': True},
        'merging_distance_threshold': {'type': float, 'positive': True}
    },
    'wsclustering': {
        'cte_traced_name': {'type': str},
        'n_clusters': {'type': dict},
        'st': {'type': float, 'positive': True},
        'ot': {'type': float, 'positive': True}
    }
}


def run_tracing(cte: ChromatinTracingExperiment, config: dict) -> ChromatinTracingExperiment:
    """ Performs the tracing on the population in parallel.

    Args:
        cte (ChromatinTracingExperiment)
        config (dict): configuration dictionary for the tracing task.

    Returns:
        cte_traced (ChromatinTracingExperiment): a new ChromatinTracingExperiment object with the traced data.
    """
    
    # Check that the tracing method is specified in the config and is valid
    if not 'method' in config:
        raise ValueError("Tracing method not specified in config.")
    if not config['method'] in tracing_available_methods:
        raise NotImplementedError("Tracing method {} not implemented.\nMust be one of: {}".
                                  format(config['method'],
                                         tracing_available_methods))
    
    def _rfunc_init(_1, _2, _3) -> dict:
        """ Initialize the traced data dictionary for the reduce function.

        Args:
            _*: not used, just to match the signature of the function

        Returns:
            traced_data (dict): empty dictionary
        """
        traced_data = {}
        return traced_data
    
    def _rfunc_update(cellID: str, traced_data: dict, cell_traced_data: dict, _1, _2) -> dict:
        """ Update the traced data dictionary for the reduce function.

        Args:
            cellID (str)
            traced_data (dict): traced data dictionary of the entire population
            cell_traced_data (dict): traced data dictionary of a single cell
            _*: not used, just to match the signature of the function

        Returns:
            traced_data (dict): updated traced data dictionary of the entire population, with the data of cellID added
        """
        traced_data[cellID] = cell_traced_data
        return traced_data
    
    # Perform parallelization: get traced data
    cte_data_traced = cte_parallel.control_func(
        cte,
        config,
        tracing_required_keys[config['method']],
        _tracing_nfunc,
        _rfunc_init,
        _rfunc_update
    )
    
    # Initialize the traced CTE object and add the traced data
    cte_traced = ChromatinTracingExperiment(config['cte_traced_name'], 'w')
    cte_traced.set_data_attrs_index(data=cte_data_traced, index=cte.index)
    
    del cte_data_traced
    
    return cte_traced

def run_tracing_single_chrom(cte: ChromatinTracingExperiment, cellID: str, chrom: str, config: dict) -> ChromatinTracingExperiment:
    """Performs a tracing algorithm on a single chromosome of a single cell.

    Args:
        cellID (str): cell ID.
        chrom (str): chromosome.
        params (dict): configuration dictionary for the Genomic Iterative DBSCAN algorithm.
    
    Returns:
        cte_chrom_traced (ChromatinTracingExperiment): a new ChromatinTracingExperiment object with the traced data.
    """
    
    # Check that 'method' is specified in the config and is valid
    if not 'method' in config:
        raise ValueError("Tracing method not specified in config.")
    if not config['method'] in tracing_available_methods:
        raise NotImplementedError("Tracing method {} not implemented.\nMust be one of: {}".
                                  format(config['method'],
                                         tracing_available_methods))
    # Check that all required keys are present in params
    cte_parallel.check_config(config, tracing_required_keys[config['method']], parallel=False)
    
    # Perform the tracing
    chrom_data = cte.get_data(cellID, chrom)
    traced_chrom_data = _chrom_tracing(chrom, chrom_data, config)
    
    # Create a new ChromatinTracingExperiment object
    cte_chrom_traced = ChromatinTracingExperiment(config['cte_traced_name'], 'w')
    
    # Add the traced data to the new ChromatinTracingExperiment object
    cte_chrom_traced.set_data_attrs_index(data={cellID: {chrom: traced_chrom_data}}, index=cte.index)
    
    del traced_chrom_data
    
    return cte_chrom_traced

def _tracing_nfunc(cellID, cte_name, config) -> dict:
    """ Node function for the tracing task.

    Args:
        cell_data (dict): data of a single cell
        config (dict): configuration dictionary for the tracing task
        _*: not used, just to match the signature of the function

    Returns:
        cell_traced_data (dict): traced data of a single cell
    """
    
    # Read the data of the cell from the HDF5 file
    with h5py.File(cte_name, 'r') as f:
        cell_data = cte_io.load_cell_data_from_hdf5(cellID, f, format='dict')
    
    # Initialize the traced data for the cell
    cell_data_traced = {}
    
    # Loop over chromosomes and perform tracing
    for chrom in cell_data:
        
        chrom_data = cell_data[chrom]
        chrom_data_traced = _chrom_tracing(chrom, chrom_data, config)
        cell_data_traced[chrom] = chrom_data_traced
        del chrom_data, chrom_data_traced
    
    return cell_data_traced

def _chrom_tracing(chrom: str, chrom_data: dict, params: dict):
    """Perform chromosome tracing on the data of a single chromosome.
    Returns the traced data of a single chromosome in dictionary format with new traceIDs.

    Args:
        chrom (str): The chromosome name.
        chrom_data (dict): The data of a single chromosome in dictionary format.
        params (dict): Parameters for GIDBSCAN.

    Returns:
        traced_chrom_data (dict): The traced data of a single chromosome in dictionary format.
    """
    
    # Convert the data to numpy arrays
    xs, ys, zs, starts, ends, lums, _, spotIDs = cte_utils.chrom_dict_to_numpy(chrom_data)
    coords = np.array([xs, ys, zs]).T
    
    # If there are less than nmin spots, return the data as is
    nmin = 20
    if len(xs) < nmin:
        return chrom_data
    
    # Perform tracing
    # GIDBSCAN
    if params['method'] == 'gidbscan':
        tracer = GenomicIterativeDBSCAN(
            params['dbscan_eps'],
            params['dbscan_min_samples'],
            params['window_size'],
            params['delta'],
            params['merging_proximity_length'],
            params['merging_overlap_threshold'],
            params['merging_distance_threshold']
            )
        tracer.fit(coords, starts)
    # Ward Spectral Clustering
    elif params['method'] == 'wsclustering':
        # Get the number of clusters
        if chrom in params['n_clusters']:
            n_clusters = params['n_clusters'][chrom]
        elif chrom.replace('chr', '').isdigit() and '#' in params['n_clusters']:  # If chrom is an autosome and '#' is in the params
            n_clusters = params['n_clusters']['#']
        else:
            raise ValueError("Number of clusters not specified for chromosome {}.".format(chrom))
        # Perform the clustering
        tracer = WardSpectralClustering(
            n_clusters,
            params['st'],
            params['ot']
        )
        tracer.fit(coords)
    # Other methods
    else:
        raise NotImplementedError("Tracing method {} not implemented.".format(params['method']))
    
    # Get the traceIDs and convert them to strings
    traceIDs = tracer.labels_.astype(str)
    
    # Convert the results back to dictionary format
    traced_chrom_data = cte_utils.chrom_numpy_to_dict(chrom, xs, ys, zs, starts, ends, lums, traceIDs, spotIDs)
    
    del xs, ys, zs, starts, ends, lums, spotIDs, coords, tracer, traceIDs
    
    return traced_chrom_data


# CLEANING

def run_sisterout_parallel(cte: ChromatinTracingExperiment, config: dict) -> dict:
    """ Run the pipeline to identify outliers in the sister chromatids.
    
    Outliers are defined as spots in a sister-sister pair (or multiplet) that are too far from each other.
    The threshold distance is specified in the configuration dictionary.

    Args:
        cte (ChromatinTracingExperiment)
        config (dict): configuration dictionary for the cleaning task.

    Returns:
        (dict): sister outliers, in a nested dictionary format:
                   outliers[cellID][chrom][traceID] = [spotID1, spotID2, ...]
    """
    
    outliers = cte_parallel.control_func(
        cte,
        config,
        sisterout_required_keys,
        sisterout_nfunc,
        sisterout_rfunc_init,
        sisterout_rfunc_update
    )
    
    return outliers

sisterout_required_keys = {
    'maxdist': {'type': float, 'positive': True},
    'neighborhood_gendist': {'type': int, 'positive': True}
}

def sisterout_nfunc(cellID: str, cte_name: str, config: dict) -> dict:
    """ Node-level function to identify outliers in the sister chromatids.

    Args:
        cellID (str)
        cte_name (str): name of the ChromatinTracingExperiment
        config (dict): configuration dictionary for the cleaning task.

    Returns:
        (dict): cell outliers in a nested dictionary format:
                 cell_outliers[chrom][traceID] = [spotID1, spotID2, ...]
    """
    
    cte = ChromatinTracingExperiment(cte_name, 'r')
    
    # Get the data of the cell
    cell_data = cte.get_data(cellID, format='dict')
    
    # Initialize set of outliers in the cell
    cell_outliers = defaultdict(dict)
    
    # Loop over chromosomes and traces
    for chrom in cell_data:
        for traceID in cell_data[chrom]:
            
            # Get the data of the trace
            trace_data = cell_data[chrom][traceID]
            
            # Convert the data to numpy arrays
            xs, ys, zs, starts, _, _, spotIDs = cte_utils.trace_dict_to_numpy(trace_data)
            crds = np.array([xs, ys, zs]).T
                 
            # Find all sister chromatids, i.e. spots with the same start position
            # it is a dictionary of numpy arrays, where:
            #   - keys are the start positions of the sisters
            #   - values are numpy arrays of indices of the spots that are sisters
            # i.e. sisters[start] = np.array([idx1, idx2, ...])
            # (only sisters with more than one spot are considered)
            sisters = find_sisters(starts)
            
            # Initialize the list of outliers for the trace
            trace_outliers = []
            
            # Loop over the sisters
            for sis_start, sis_idx in sisters.items():
                
                # Get the positions and spotIDs of the sister chromatids
                crds_sis = crds[sis_idx, :]
                spotIDs_sis = spotIDs[sis_idx]
                
                # Get the median neighborhood of the sisters,
                # i.e. the median position of all the spots
                # within a certain genomic distance (specified in config) from the sister
                neigh_gendist = config['neighborhood_gendist']
                med = get_median_neighborhood(sis_start, crds, starts, neigh_gendist)
                
                # Find the outliers
                maxdist = config['maxdist']  # maximum distance between sisters not to be considered an outlier
                outliers = clean_sister(crds_sis, spotIDs_sis, med, maxdist)
                
                # Add the outliers to the set of spots to remove
                if len(outliers) == 0:
                    continue
                trace_outliers.extend(outliers)
                
                del crds_sis, spotIDs_sis, med, outliers
            
            # Add the outliers of the trace to the outliers of the cell
            if len(trace_outliers) == 0:
                continue
            cell_outliers[chrom][traceID] = trace_outliers
            
            del trace_data, xs, ys, zs, starts, sisters

    del cell_data
    cte.close()
    
    return cell_outliers

def find_sisters(starts: np.ndarray) -> dict:
    """ Find the sister chromatids in a list of start positions.

    Args:
        starts (np.ndarray): array of start positions of the spots in a trace.

    Returns:
        (dict): a dictionary of numpy arrays of indices of the spots that are sisters,
                where the keys are the start positions of the sisters:
                sisters[start] = np.array([idx1, idx2, ...])
    """
    # Find the unique start positions and their counts
    unique_starts, counts = np.unique(starts, return_counts=True)
    # Remove the unique starts with only one spot
    unique_starts = unique_starts[counts > 1]
    # Get the sisters, i.e. a dict of numpy arrays of indices of the spots that are sisters
    sisters = {}
    for start in unique_starts:
        sisters[start] = np.where(starts == start)[0]
    return sisters

def get_median_neighborhood(sis_start: int, crds: np.ndarray, starts: np.ndarray, neigh_gendist: int) -> np.ndarray:
    """ Get the median neighborhood of the sister chromatids.

    Args:
        start (int): start position of the sister chromatids.
        xs (np.ndarray): array of x-coordinates of the spots in the trace.
        ys (np.ndarray): array of y-coordinates of the spots in the trace.
        zs (np.ndarray): array of z-coordinates of the spots in the trace.
        starts (np.ndarray): array of start positions of the spots in the trace.
        neigh_gendist (int): maximum genomic distance for a spot to be considered in the neighborhood of the sister chromatids.

    Returns:
        (np.ndarray): array of shape (3,) of the median neighborhood of the sister chromatids.
    """
    # Get the mask of the spots in the neighborhood
    mask = np.abs(starts - sis_start) <= neigh_gendist
    # Remove the sister spots from the neighborhood mask
    mask[starts == sis_start] = False
    # If there are no spots in the neighborhood, return None (no median point)
    if np.sum(mask) == 0:
        return None
    # Get the coordinates of the spots in the neighborhood
    crds_neigh = crds[mask, :]
    # Get the centroid of the entire trace
    centroid = np.mean(crds, axis=0)
    # Get the index of the median neighborhood of the sister chromatids
    med_idx = utils.spots_3d_median(crds_neigh, centroid)
    return crds_neigh[med_idx]

def clean_sister(crds: np.ndarray, spotIDs: np.ndarray, med: np.ndarray, maxdist: float) -> list:
    """ Clean the sister chromatids, removing the outliers.

    Args:
        crds (np.ndarray): array of shape (n, 3) of the coordinates of the sister chromatids.
        spotIDs (np.ndarray): array of shape (n,) of the spotIDs of the sister chromatids.
        med (np.ndarray): array of shape (3,) of the median neighborhood of the sister chromatids.
        maxdist (float): maximum distance for a spot to be considered an outlier.

    Returns:
        (list): list of spotIDs that are outliers.
    """
    
    # Initialize the list of outliers
    outliers = []
    
    # Loop until all outliers are removed
    while True:
    
        # Remove the data (from crds and spotIDs) of the outliers
        mask = np.isin(spotIDs, outliers, invert=True)  # True for spots to keep
        crds_m = crds[mask, :]
        spotIDs_m = spotIDs[mask]
        
        # If there are either no spots or only one spot, exit the loop
        if len(spotIDs_m) < 2:
            break
        
        # Get the distances between the masked spots
        dists_m = cdist(crds_m, crds_m)            
        # Set the diagonal to NaN (avoid self-comparison)
        np.fill_diagonal(dists_m, np.nan)
        # Set the lower triangle to NaN (avoid double comparison)
        dists_m[np.tril_indices(dists_m.shape[0])] = np.nan
        
        # If the maximum distance between the spots is less than the threshold, exit the loop
        if np.nanmax(dists_m) < maxdist:
            break
        
        # Otherwise, remove one or more spots
        
        # If there are only two spots:
        #  - if the med point is not provided (it's None), remove both spots
        #  - if the med point is provided AND both spots are close to the med point, remove both spots
        #  - if the med point is provided AND both spots are too far from the med point, remove both spots
        #  - if the med point is provided AND only one of the spots is too far from the med point, remove this spot
        if len(spotIDs_m) == 2:
            # Case 1: med point is not provided
            if med is None:
                outliers.extend(spotIDs_m)  # add both spots to the outliers
            else:
                # If med exists, get the distances between the spots and the med point
                dists_to_med = cdist(crds_m, np.array([med]))  # med has to be reshaped to (1, 3)
                # Case 2: med point is provided and both spots are close to the med point
                if np.max(dists_to_med) < maxdist:
                    outliers.extend(spotIDs_m)
                # Case 3: med point is provided and both spots are too far from the med point
                elif np.min(dists_to_med) > maxdist:
                    outliers.extend(spotIDs_m)
                # Case 4: med point is provided and only one of the spots is too far from the med point
                else:
                    idx = np.argmax(dists_to_med)
                    outliers.append(spotIDs_m[idx])
        
        # If there are 3 or more spots, remove the one with the largest distance to the others
        else:
            sum_dists = np.nansum(dists_m, axis=0)
            max_idx = np.nanargmax(sum_dists)
            outliers.append(spotIDs_m[max_idx])
    
    return outliers

def sisterout_rfunc_init(_1, _2, _3) -> dict:
    """ Initialize the outliers dictionary for the reduce function.

    Args:
        _*: not used, just to match the signature of the function
    
    Returns:
        (dict): empty dictionary to store all outliers
    """
    return {}

def sisterout_rfunc_update(cellID: str, outliers: dict, cell_outliers: dict, _1, _2) -> dict:
    """ Update the outliers dictionary for the reduce function.
    Adds the outliers of a single cell to the dictionary of all outliers.

    Args:
        cellID (str)
        outliers (dict): dictionary of all outliers
        cell_outliers (dict): dictionary of the outliers of a single cell
        _*: not used, just to match the signature of the function

    Returns:
        (dict): updated dictionary of all outliers
    """
    outliers[cellID] = cell_outliers
    return outliers


# PROJECTION

def run_projection(cte: ChromatinTracingExperiment, config: dict) -> ChromatinTracingExperiment:
    """ Performs the "projection" of the CTE data to a target resolution.
    
    It consists of coarse-graining the data to a target resolution, where each spot is mapped to a single domain.
    When multiple spots are mapped to the same domain, the center of mass is calculated.
    
    The target resolution is specified in the configuration dictionary.
    
    This function will create a new ChromatinTracingExperiment object with the projected data,
    where only one spot per coarsed domain is kept.

    Args:
        cte (ChromatinTracingExperiment)
        config (dict): configuration dictionary for the projection task.

    Returns:
        ChromatinTracingExperiment: a new ChromatinTracingExperiment object with the projected data.
    """
    
    # Coarse-grain the index of the CTE to the target resolution
    res = config['resolution']
    if res == 'self':
        index_prj = deepcopy(cte.index)
    elif isinstance(res, int):
        index_prj = cte.index.coarsegrain(res)
    else:
        raise ValueError("Invalid target resolution: {}".format(res))
    
    # Get the data of the projected CTE (dictionary format)
    data_prj = cte_parallel.control_func(
        cte,
        config,
        projection_required_keys,
        projection_nfunc,
        projection_rfunc_init,
        projection_rfunc_update
    )
    
    # Create the projected CTE object
    cte_prj_h5name = cte.h5_name.replace('.h5', '_projected.h5')
    cte_prj = ChromatinTracingExperiment(cte_prj_h5name, 'w')
    cte_prj.set_data_attrs_index(data=data_prj, index=index_prj)
    
    # If the original CTE has a cell_states group, copy it to the projected CTE
    if 'cell_states' in cte:
        cte_prj.set_cell_states(cte.cell_states)
    
    # If the original CTE has an alphashape group, copy it to the projected CTE
    if 'alphashapes' in cte:
        cte_prj.set_alphashapes(cte.get_alphashapes())
    
    del data_prj
    
    return cte_prj

projection_required_keys = {
    'resolution': {'type': [int, str]}
}

def projection_nfunc(cellID: str, cte_name: str, config: dict) -> dict:
    """ Node-level function to perform the projection of the CTE data to a target resolution on a single cell.
    
    The function first coarse-grains the index of the CTE,
    then reads through the data and maps each spot (for each chrom/trace) to the coarse-grained domain.
    Finally, it calculates the center of mass whenever multiple spots are mapped to the same domain, creating the projected data.

    Args:
        cellID (str)
        cte_name (str)
        config (dict): configuration dictionary for the projection task

    Returns:
        dict: projected data of the cell in dictionary format
    """
    
    # Read the CTE, get the cell data and the index
    cte = ChromatinTracingExperiment(cte_name, 'r')
    cell_data = cte.get_data(cellID, format='dict')
    index = cte.index
    
    # Coarse-grain the index to the target resolution
    res = config['resolution']
    if res == 'self':
        index_coarse = deepcopy(index)
    else:
        index_coarse = index.coarsegrain(res)
    
    # Map the indices from the original index to the coarse-grained index, e.g.
    #    map_to_coarse = {
    #           ('chr1', 100000, 125000): [('chr1', 100000, 150000)],
    #           ('chr1', 125000, 150000): [('chr1', 100000, 150000)],
    #           ('chr1', 150000, 175000): [('chr1', 150000, 200000)],
    #           ...
    #       }
    map_to_coarse = map_indices(index, index_coarse)
    
    # Check that the mapping is correct:    
    # 1) the length of the mapping is the same as the length of the original index
    assert len(map_to_coarse) == len(index), "Length of the mapping does not match the length of the original index."
    for dom in map_to_coarse:
        domcoarse = map_to_coarse[dom]
        # 2) each domain in the original index maps to exactly one domain in the coarse-grained index
        assert isinstance(domcoarse, list), "The domain does not map to a list."
        assert len(domcoarse) == 1, "Multiple domains map to the same coarse-grained domain."
        # 3) the chromosomes of the original and coarse-grained indices match
        assert domcoarse[0][0] == dom[0], "Chromosomes of the original and coarse-grained indices do not match."
        # 4) The start/end of the coarse-grained domain includes the start/end of the original domain
        assert domcoarse[0][1] <= dom[1], "Start positions of the original and coarse-grained indices do not match."
        assert domcoarse[0][2] >= dom[2], "End positions of the original and coarse-grained indices do not match."
    
    # Change the dictionary values from lists of one element to the element itself
    for dom in map_to_coarse:
        map_to_coarse[dom] = map_to_coarse[dom][0]
    
    # Initialize a dictionary to store the cell data indexed by the coarse-grained domains
    # It's going to be a nested dictionary of the form:
    #   data_by_domcoarse = {
    #       'chr1': {
    #           'trace1': {
    #               ('chr1', 100000, 150000): {   (coarse-grained domain)
    #                   'xs': [x1, x2, ...],      (x coordinates of all spots mapped to the domain for this chrom/trace)
    #                   'ys': [y1, y2, ...],
    #                   'zs': [z1, z2, ...],
    #                   'lums': [lum1, lum2, ...]
    #                                         }
    #                   ...
    #                     }
    #              ...  
    #               } 
    #          ...       
    #                       }
    data_by_domcoarse = {}
    
    # Loop over chrom/trace/spot and map the spot data to the coarse-grained domain
    for chrom in cell_data:
        for traceID in cell_data[chrom]:
            for spotID in cell_data[chrom][traceID]:
                
                # Get the spot data
                spot_data = cell_data[chrom][traceID][spotID]
                
                # Unpack the spot data
                x, y, z = spot_data['x'], spot_data['y'], spot_data['z']
                start, end = spot_data['start'], spot_data['end']
                lum = spot_data['lum']
                
                # Get the coarse-grained domain of the spot (chrom, start_coarse, end_coarse)
                domcoarse = map_to_coarse[(chrom, start, end)]
                
                # Add the spot to the data indexed by the coarse-grained domain
                if chrom not in data_by_domcoarse:
                    data_by_domcoarse[chrom] = {}
                if traceID not in data_by_domcoarse[chrom]:
                    data_by_domcoarse[chrom][traceID] = {}
                if domcoarse not in data_by_domcoarse[chrom][traceID]:
                    data_by_domcoarse[chrom][traceID][domcoarse] = {
                        'xs': [], 'ys': [], 'zs': [], 'lums': []
                    }
                data_by_domcoarse[chrom][traceID][domcoarse]['xs'].append(x)
                data_by_domcoarse[chrom][traceID][domcoarse]['ys'].append(y)
                data_by_domcoarse[chrom][traceID][domcoarse]['zs'].append(z)
                data_by_domcoarse[chrom][traceID][domcoarse]['lums'].append(lum)
    
    # Loop over the data indexed by the coarse-grained domain: when there are multiple spots in a domain, calculate the center of mass
    cell_data_prj = {}
    # Initialize the number of spots to create new unique spotIDs by order of appearance
    nspot = 1
    for chrom in data_by_domcoarse:
        for traceID in data_by_domcoarse[chrom]:
            for domcoarse in data_by_domcoarse[chrom][traceID]:
                
                # Add chrom/traceID to cell_data_prj if not already present
                if chrom not in cell_data_prj:
                    cell_data_prj[chrom] = {}
                if traceID not in cell_data_prj[chrom]:
                    cell_data_prj[chrom][traceID] = {}
                
                # Get the spotID
                spotID = str(nspot)
                nspot += 1
                
                # Get the data of the coarse domain
                xs = data_by_domcoarse[chrom][traceID][domcoarse]['xs']
                ys = data_by_domcoarse[chrom][traceID][domcoarse]['ys']
                zs = data_by_domcoarse[chrom][traceID][domcoarse]['zs']
                lums = data_by_domcoarse[chrom][traceID][domcoarse]['lums']
                
                # Calculate the center of mass and the average luminescence
                com = np.mean(np.array([xs, ys, zs]).T, axis=0)
                lum = np.mean(lums)
                
                # Create the spot data
                spot_data = {
                    'x': com[0],
                    'y': com[1],
                    'z': com[2],
                    'chrom': chrom,
                    'start': domcoarse[1],
                    'end': domcoarse[2],
                    'lum': lum
                }
                
                # Add the spot data to the cell_data_prj dictionary
                cell_data_prj[chrom][traceID][spotID] = spot_data
    
    del cell_data, index, index_coarse, map_to_coarse, data_by_domcoarse
    cte.close()
    
    return cell_data_prj

def projection_rfunc_init(_1, _2, _3) -> dict:
    """ Initialize the projected data dictionary for the reduce function.

    Args:
        _*: not used, just to match the signature of the function

    Returns:
        (dict): empty dictionary to store the population projected data
    """
    return {}

def projection_rfunc_update(cellID: str, data_prj: dict, cell_data_prj: dict, _1, _2) -> dict:
    """ Update the projected data dictionary for the reduce function.
    Adds the projected data of a single cell to the dictionary of the population projected data.

    Args:
        cellID (str)
        data_prj (dict): dictionary of the projected data of the entire population
        cell_data_prj (dict): dictionary of the projected data of a single cell
        _*: not used, just to match the signature of the function

    Returns:
        (dict): updated dictionary of all the projected data
    """
    data_prj[cellID] = cell_data_prj
    return data_prj



def _OLD_run_cleaning(cte: ChromatinTracingExperiment, coverage_threshold: float, gendist_threshold: float) -> ChromatinTracingExperiment:
    """ Performs the cleaning of the traced data.
    
    Creates a new ChromatinTracingExperiment object with the cleaned data, i.e. without:
        - noisy traces
        - traces with a too-low coverage (less than coverage_threshold)
        - traces with a too-large minimum genomic distance between neighbors

    Args:
        cte (ChromatinTracingExperiment)
        coverage_threshold (float): minimum coverage for a trace to be kept.
        gendist_threshold (float): maximum threshold for the minimum genomic distance between neighbors for a trace to be kept.

    Returns:
        cte_clean (ChromatinTracingExperiment): a new ChromatinTracingExperiment object with the cleaned data.
    """
    
    # Initialize the cleaned data
    clean_data = {}
    
    # Loop over cells, chromosomes and traces and fill lists
    for cellID in cte.data:
        clean_data[cellID] = {}  # initialize dictionary for cellID
        
        for chrom in cte.data[cellID]:
            clean_data[cellID][chrom] = {}  # initialize dictionary for chrom
            
            for traceID in cte.data[cellID][chrom]:
                    
                    # Ignore noisy traces
                    if cte.look_for_noisy_trace(traceID):
                        continue
                    
                    # Ignore traces with low coverage
                    coverage = metrics.compute_trace_coverage(cte, cellID, chrom, traceID)
                    if coverage < coverage_threshold:
                        continue
                    
                    # Compute the minimum genomic distance between neighboring spots
                    gdist, _ = metrics.compute_trace_neighbor_distances(cte, cellID, chrom, traceID)
                    min_gdist = np.min(gdist)
                    if min_gdist > gendist_threshold:
                        continue
                    
                    # If everything is ok, add the trace to the cleaned data
                    clean_data[cellID][chrom][traceID] = cte.data[cellID][chrom][traceID]
            
            # If chrom data is empty, delete it
            if clean_data[cellID][chrom] == {}:
                del clean_data[cellID][chrom]
        
        # If cell data is empty, delete it
        if clean_data[cellID] == {}:
            del clean_data[cellID]
    
    # Create a new ChromatinTracingExperiment object
    cte_clean = ChromatinTracingExperiment()
    
    # Add the traced data to the new ChromatinTracingExperiment object
    cte_clean.set_data_attrs_index(data=clean_data, assembly=cte.assembly, index=cte.index)
    
    del clean_data
    
    return cte_clean
