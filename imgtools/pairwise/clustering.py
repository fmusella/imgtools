import os
import h5py
import numpy as np
from sklearn.cluster import DBSCAN
from ..cte import ChromatinTracingExperiment
from ..scf import SingleCellFeature
from .. import parallel


# FUNCTIONS TO SELECT SPOTS, RANDOMIZE AND CLUSTERING

def get_spots_mask(
    cellID: str, config: dict,
    cte: ChromatinTracingExperiment, scf_name: str,
) -> np.ndarray:
    """ Get the mask to select spots of interest.
    
    There are two methods to select spots of interest:
        1. 'by_SCF': Select spots based on a feature from the SingleCellFeature (SCF).
            Both a feature and a percentile must be provided, and the spots
            are selected based on whether their feature value is above the percentile.
        2. 'by_BED': Select spots based on a BED file.
            Spots are selected based on whether they are present in the BED file.
            Note: the BED file must have the same Index as the CTE file. The domains
            should be indicated with a 0 / 1 value in the BED file.

    Args:
        cellID (str)
        config (dict)
        cte (ChromatinTracingExperiment)
        scf_name (str)

    Returns:
        mask (np.ndarray): A boolean mask indicating which spots are selected.
    """
    
    # Make sure that the spots selection method is valid
    accepted_methods = ['by_SCF', 'by_BED']
    if not config['spots_selection_method'] in accepted_methods:
        raise ValueError(
            f"Invalid spots selection method: {config['spots_selection_method']}. "
            f"Accepted methods are: {accepted_methods}"
        )
    
    # Get the mask for the SCF selection method
    if config['spots_selection_method'] == 'by_SCF':
        
        # Make sure that the additional keys are present in the config
        additional_keys = ['feature', 'feat_percentile']
        for key in additional_keys:
            if key not in config:
                raise KeyError(f"Missing required key: {key} in config")
        
        # Open the SCF file
        scf = SingleCellFeature(scf_name, 'r')
        
        # Read the feature data sorted by spotIDs
        featvals = scf.get_feature_by_spotIDs(cellID, cte, config['feature'])
        
        mask = featvals >= np.nanpercentile(featvals, config['feat_percentile'])
    
    # Get the mask for the BED selection method
    elif config['spots_selection_method'] == 'by_BED':
        
        # Make sure that the additional keys are present in the config
        additional_keys = ['BED']
        for key in additional_keys:
            if key not in config:
                raise KeyError(f"Missing required key: {key} in config")
        
        # Read the BED file in the same order as the spots in the CTE
        try:
            mask = cte.get_bed_values_by_spotIDs(cellID, config['BED'])
            mask = mask.astype(bool)
        except Exception as e:
            raise ValueError(
                f"Error reading BED file '{config['BED']}' for cell '{cellID}': {e}"
            )
    
    return mask

def randomize_spots_mask():
    pass

def cluster_spots(xs, ys, zs, mask, config):
    """
    Here I can write down most of the code that is currently in node_function.
    
    I just give the coordinates and the mask, together with the config,
    and perform the clustering.
    
    This is independent on whether the calculation is for the actual mask,
    or for the randomized one.
    
    Then in node_function I can have two conditions for the case and the random control.
    The main difference is that I want to add - only for the randomization - the option
    to repeat the randomization multiple times in the same cell.
    
    So I just call this function in node_function.
    
    """
    pass


# FUNCTIONS TO PARALLELIZE THE CLUSTERING CALCULATION

def node_function(cellID: str, cte_name: str, scf_name: str, config: dict) -> dict:
    """ Node-level function to perform clustering on a single cell.
    
    This function reads the data for a cell, selects the spots of interest,
    and performs clustering on the selected spots.
    
    Right now, only DBSCAN clustering is implemented.
    
    The results are returned as a dictionary containing:
        - Number of clusters
        - Number of spots per cluster
        - Points in each cluster
        - Center of mass for each cluster
        - RMSD for each cluster

    Args:
        cellID (str)
        cte_name (str)
        scf_name (str)
        config (dict): configuration dictionary containing:
            - eps (float): The maximum distance between two samples for one
                    to be considered as in the neighborhood of the other.
            - min_samples (int): The number of samples (or total weight) in
                    a neighborhood for a point to be considered as a core point.
            - h5_file (str): Path to the output HDF5 file where results will be stored.

    Returns:
        dict: A dictionary containing the clustering results for the cell.
    """
    
    # Read the CTE file
    cte = ChromatinTracingExperiment(cte_name, 'r')
    
    # Get the data in numpy format
    xs, ys, zs, chroms, starts, ends, _, _, _ = cte.get_data(cellID, format='numpy')
    
    # Get the mask for the spots of interest
    mask = get_spots_mask(cellID, config, cte, scf_name)
    
    # Filter the data based on the mask
    xs, ys, zs = xs[mask], ys[mask], zs[mask]
    chroms, starts, ends = chroms[mask], starts[mask], ends[mask]
    
    # Perform DBSCAN clustering and fit to the data
    dbscan = DBSCAN(eps=config['eps'], min_samples=config['min_samples'])
    dbscan.fit(np.column_stack((xs, ys, zs)))
    # Get the labels from the clustering
    labels = dbscan.labels_
    
    # Initialize a dictionary to hold the results
    results = {}
    
    # Store the number of clusters
    results['n_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Store the number of spots per cluster
    results['n_spots_per_cluster'] = {}
    for l in set(labels):
        if l == -1:  # Skip noise cluster
            continue
        results['n_spots_per_cluster'][l] = np.sum(labels == l)
    
    # Store the data of the points in each cluster,
    # their center of mass, and RMSD
    results['points'] = {}
    results['center'] = {}
    results['rmsd'] = {}
    for l in set(labels):
        if l == -1:  # Skip noise cluster
            continue
        # Mask for the current cluster
        mask_l = labels == l
        xs_l, ys_l, zs_l = xs[mask_l], ys[mask_l], zs[mask_l]
        # Store the points
        results['points'][l] = {
            'xs': xs_l, 'ys': ys_l, 'zs': zs_l,
            'chroms': chroms[mask_l],
            'starts': starts[mask_l],
            'ends': ends[mask_l]
        }
        # Calculate the center of mass
        x_com, y_com, z_com = np.mean(xs_l), np.mean(ys_l), np.mean(zs_l)
        # Store the center of mass
        results['center'][l] = np.array([x_com, y_com, z_com])
        # Calculate the RMSD
        rmsd = np.sqrt(np.mean((xs_l - x_com)**2 + (ys_l - y_com)**2 + (zs_l - z_com)**2))
        # Store the RMSD
        results['rmsd'][l] = rmsd
    
    # Close the CTE file
    cte.close()
    
    return results

def reduce_initialization(cellIDs: list, cte_name: str, scf_name: str, config: dict) -> h5py.File:
    """ Initializes the h5py file to store the results of each cell clustering.

    Args:
        cellIDs (list)
        cte_name (str)
        scf_name (str)
        config (dict)

    Returns:
        h5py.File: An initialized h5py file to store clustering results.
    """
    
    # Initialize the h5py file to store the results of each cell clustering
    h5_file = config['h5_file']
    cluster_h5 = h5py.File(h5_file, 'w')
    
    return cluster_h5

def reduce_update(cellID: str, cluster_h5: h5py.File, cell_result: dict, cte_name: str, scf_name: str, config: dict) -> h5py.File:
    """ Updates the h5py file with clustering results for a single cell.
    
    This function creates a group for the cellID in the h5py file and stores:
        - Number of clusters (as an attribute)
        - For each cluster (group):
          - Number of spots in the cluster (as an attribute)
          - Points in the cluster (subgroup with datasets):
            - xs (as a dataset)
            - ys (as a dataset)
            - zs (as a dataset)
            - chroms (as a dataset)
            - starts (as a dataset)
            - ends (as a dataset)
          - Center of mass (as a dataset)
          - RMSD (as a dataset)
        

    Args:
        cellID (str): _description_
        cluster_h5 (h5py.File): _description_
        cell_result (dict): _description_
        cte_name (str): _description_
        scf_name (str): _description_
        config (dict): _description_

    Returns:
        h5py.File: The updated h5py file with clustering results for the cell.
    """
    
    # Create a group for the cellID
    cell_group = cluster_h5.create_group(cellID)
    
    # Store the number of clusters as an attribute
    cell_group.attrs['n_clusters'] = cell_result['n_clusters']
    
    # Loop through the unique cluster labels
    for l in cell_result['points'].keys():
        if l == -1:  # Skip noise cluster
            continue
        # Create a subgroup for the cluster
        cluster_group = cell_group.create_group(str(l))
        # Store the number of spots in the cluster as an attribute
        cluster_group.attrs['n_spots'] = cell_result['n_spots_per_cluster'][l]
        # Store the points (as a subgroup)
        points_group = cluster_group.create_group('points')
        points_group.create_dataset('xs', data=cell_result['points'][l]['xs'])
        points_group.create_dataset('ys', data=cell_result['points'][l]['ys'])
        points_group.create_dataset('zs', data=cell_result['points'][l]['zs'])
        points_group.create_dataset('chroms', data=cell_result['points'][l]['chroms'].astype('S'))
        points_group.create_dataset('starts', data=cell_result['points'][l]['starts'])
        points_group.create_dataset('ends', data=cell_result['points'][l]['ends'])
        # Store the center of mass
        cluster_group.create_dataset('center', data=cell_result['center'][l])
        # Store the RMSD
        cluster_group.create_dataset('rmsd', data=cell_result['rmsd'][l])
    
    # Return the updated h5py file
    return cluster_h5


# MAIN FUNCTION TO RUN THE CLUSTERING

# Define the required keys for the configuration dictionary
required_keys = {
    'spots_selection_method': {'type': str},
    'eps': {'type': float, 'positive': True},
    'min_samples': {'type': int, 'positive': True},
    'h5_file': {'type': str},
}

def clustering(
    cte: ChromatinTracingExperiment, scf: SingleCellFeature, config: dict
) -> h5py.File:
    """ Performs clustering of selected spots of interest in each cell of a
    Chromatin Tracing Experiment.
    
    The clustering is performed using the DBSCAN algorithm.
    
    The results are stored in an HDF5 file with the following structure:
    - /cellID/n_clusters (attribute)
    - /cellID/cluster_label/n_spots (attribute)
    - /cellID/cluster_label/points/xs (dataset)
    - /cellID/cluster_label/points/ys (dataset)
    - /cellID/cluster_label/points/zs (dataset)
    - /cellID/cluster_label/points/chroms (dataset)
    - /cellID/cluster_label/points/starts (dataset)
    - /cellID/cluster_label/points/ends (dataset)
    - /cellID/cluster_label/center (dataset)
    - /cellID/cluster_label/rmsd (dataset)
    
    This function uses parallel processing to perform clustering on each cell.

    Args:
        cte (ChromatinTracingExperiment)
        scf (SingleCellFeature or None): the SCF is only required if the spots
            of interest are selected based on the SCF. Otherwise, it can be None.
        config (dict): configuration dictionary containing:
            - eps (float): The maximum distance between two samples for one
                    to be considered as in the neighborhood of the other.
            - min_samples (int): The number of samples (or total weight) in
                    a neighborhood for a point to be considered as a core point.
            - h5_file (str): Path to the output HDF5 file where results will be stored.

    Returns:
        h5py.File: An HDF5 file containing clustering results for each cell.
    """
    
    cluster_h5 = parallel.control_func(
        cte, scf, config, required_keys,
        node_function, reduce_initialization, reduce_update
    )
    
    return cluster_h5