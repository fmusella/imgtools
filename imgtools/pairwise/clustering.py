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

def randomize_spots_mask(mask: np.ndarray, chroms: np.ndarray) -> np.ndarray:
    """ Randomize the mask for the spots of interest.
    
    The randomization preserves the total number of True values across the genome.
    Each chromosome is randomly mapped to another chromosome, and then loci in
    the mapped chromosome are randomly selected to match the number of Trues
    in the original chromosome.

    Args:
        mask (np.ndarray): original boolean mask for the spots of interest.
        chroms (np.ndarray): chromosome identifiers for each spot.

    Returns:
        np.ndarray: randomized boolean mask for the spots of interest.
    """
    
    # Randomly map the chromosomes to each other
    unique_chroms = np.unique(chroms)
    unique_chroms_perm = np.random.permutation(unique_chroms)  # permute the chromosomes
    # Make sure that there is no identity mapping
    while np.any(unique_chroms == unique_chroms_perm):
        unique_chroms_perm = np.random.permutation(unique_chroms)
    chroms_map = {unique_chroms[i]: unique_chroms_perm[i] for i in range(len(unique_chroms))}
    
    # Initialize the randomized mask
    mask_rand = np.full(mask.shape, False, dtype=bool)
    
    # Loop over the chromosomes and randomize the values
    for chrom, chrom_r in chroms_map.items():
        
        # Get the number of Trues in the original mask for this chromosome
        ntrue = np.sum(mask[chroms == chrom])
        
        # If there are no Trues, continue
        if ntrue == 0:
            continue
        
        # If the number of Trues is greater than the number of spots in
        # the randomized chromosome, raise an error
        nspot_chrom_r = np.sum(chroms == chrom_r)
        if ntrue > nspot_chrom_r:
            raise ValueError(f'Cannot randomize: {chrom} has {ntrue} Trues, {chrom_r} has {nspot_chrom_r} spots.')
        
        # Create a random selection of indices in the randomized chromosome
        indices_r = np.random.choice(nspot_chrom_r, size=ntrue, replace=False)
        
        # Set the corresponding indices in the randomized mask to True
        mask_rand_chrom_r = mask_rand[chroms == chrom_r]
        mask_rand_chrom_r[indices_r] = True
        mask_rand[chroms == chrom_r] = mask_rand_chrom_r
    
    return mask_rand

def cluster_spots(
    xs: np.ndarray, ys: np.ndarray, zs: np.ndarray,
    chroms: np.ndarray, starts: np.ndarray, ends: np.ndarray,
    eps: float, min_samples: int
):
    """ Cluster the spots using DBSCAN.
    
    The results are returned as a dictionary containing:
    - Number of clusters
    - Number of spots per cluster
    - Points in each cluster
    - Center of mass for each cluster
    - RMSD for each cluster

    Args:
        xs (np.ndarray): x coordinates of the spots,
        ys (np.ndarray): y coordinates of the spots,
        zs (np.ndarray): z coordinates of the spots,
        chroms (np.ndarray): chromosome identifiers of the spots,
        starts (np.ndarray): start positions of the spots,
        ends (np.ndarray): end positions of the spots,
        eps (float): DBSCAN eps parameter, minimum distance between points
            to be considered in the same neighborhood.
        min_samples (int): DBSCAN min_samples parameter, minimum number of
            points to form a dense region.

    Returns:
        dict: A dictionary containing the clustering results. Keys are:
            - 'n_clusters': int, number of clusters found
            - 'n_spots_per_cluster': dict, number of spots per cluster
            - 'points': dict, points in each cluster
            - 'center': dict, center of mass for each cluster
            - 'rmsd': dict, RMSD for each cluster
    """
    
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
    
    # Perform DBSCAN clustering and fit to the data
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
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
    
    return results


# FUNCTIONS TO PARALLELIZE THE CLUSTERING CALCULATION

def node_function(cellID: str, cte_name: str, scf_name: str, config: dict) -> dict:
    """ Node-level function to perform clustering on a single cell.
    
    This function reads the data for a cell, selects the spots of interest,
    and performs clustering on the selected spots.
    
    Right now, only DBSCAN clustering is implemented.
    
    If the config contains the key 'n_randomizations' with a value > 0,
    this function also calculates clusters by randomizing the spots of interest.
    
    The results are returned as a dictionary containing:
        - For the observed data:
            - Number of clusters,
            - Number of spots per cluster,
            - Points in each cluster,
            - Center of mass for each cluster,
            - RMSD for each cluster.
        - For each randomization (if requested):
            - Number of clusters,
            - Number of spots per cluster.

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
            Contains two keys: 'observed' and 'nulls'.
            'observed' contains the following keys:
                - 'n_clusters': int, number of clusters found,
                - 'n_spots_per_cluster': dict, number of spots per cluster,
                - 'points': dict, points in each cluster,
                - 'center': dict, center of mass for each cluster,
                - 'rmsd': dict, RMSD for each cluster.
            'nulls' contains a dictionary of randomization results.
                If no randomizations are performed, this will be empty.
                Otherwise, contains keys for each randomization index, each with:
                - 'n_clusters': int, number of clusters found,
                - 'n_spots_per_cluster': dict, number of spots per cluster.
    """
    
    # Read the CTE file
    cte = ChromatinTracingExperiment(cte_name, 'r')
    
    # Get the data in numpy format
    d = cte.get_data(cellID, format='numpy')
    xs, ys, zs, chroms, starts, ends = d['xs'], d['ys'], d['zs'], d['chroms'], d['starts'], d['ends']
    
    # Get the mask for the spots of interest
    mask = get_spots_mask(cellID, config, cte, scf_name)
    
    # Filter the data based on the mask
    xs_m, ys_m, zs_m = xs[mask], ys[mask], zs[mask]
    chroms_m, starts_m, ends_m = chroms[mask], starts[mask], ends[mask]
    
    # Initialize the results dictionary
    results = {'observed': {}, 'nulls': {}}
    
    # Perform DBSCAN clustering and fit to the data
    results['observed'] = cluster_spots(
        xs_m, ys_m, zs_m, chroms_m, starts_m, ends_m, config['eps'], config['min_samples']
    )
    
    # If no randomization is requested, return the results
    if 'n_randomizations' not in config or config['n_randomizations'] == 0:
        cte.close()
        return results
    
    # Otherwise, perform randomizations
    for i in range(config['n_randomizations']):
        
        # Randomize the mask for the spots
        mask_rnd = randomize_spots_mask(mask, chroms)
        
        # Filter the data based on the randomized mask
        xs_r, ys_r, zs_r = xs[mask_rnd], ys[mask_rnd], zs[mask_rnd]
        chroms_r, starts_r, ends_r = chroms[mask_rnd], starts[mask_rnd], ends[mask_rnd]
        
        # Perform DBSCAN clustering on the randomized data
        results_rnd_i = cluster_spots(
            xs_r, ys_r, zs_r, chroms_r, starts_r, ends_r, config['eps'], config['min_samples']
        )
        # Remove the 'points', 'center' and 'rmsd' keys to save memory
        del results_rnd_i['points'], results_rnd_i['center'], results_rnd_i['rmsd']
        
        # Store the results for this randomization
        results['nulls'][i] = results_rnd_i
    
    cte.close()
    return results

def reduce_initialization(_1, _2, _3, config: dict) -> h5py.File:
    """ Initializes the h5py file to store the results of each cell clustering.

    Args:
        config (dict)
        _*: not used, just to match the signature of the function.

    Returns:
        h5py.File: An initialized h5py file to store clustering results.
    """
    
    # Initialize the h5py file to store the results of each cell clustering
    h5_file = config['h5_file']
    cluster_h5 = h5py.File(h5_file, 'w')
    
    return cluster_h5

def reduce_update(cellID: str, cluster_h5: h5py.File, cell_result: dict, _1, _2, _3) -> h5py.File:
    """ Updates the h5py file with clustering results for a single cell.
    
    The structure of the h5py file is as follows:
    - cellID/
    -   observed/
    -      n_clusters (attribute)
    -      cluster_label/ (group)
    -         n_spots (attribute)
    -         points/ (group)
    -            xs (dataset)
    -            ys (dataset)
    -            zs (dataset)
    -            chroms (dataset)
    -            starts (dataset)
    -            ends (dataset)
    -         center (dataset)
    -         rmsd (dataset)
    -   nulls/
    -      randomization_index/ (group)
    -         n_clusters (attribute)
    -         n_spots_cluster_label (attribute for each cluster)

    Args:
        cellID (str)
        cluster_h5 (h5py.File): The h5py file to update.
        cell_result (dict): The clustering results for the cell.
        _*: not used, just to match the signature of the function.

    Returns:
        h5py.File: The updated h5py file with clustering results for the cell.
    """
    
    # Create a group for the cellID
    cell_group = cluster_h5.create_group(cellID)
    
    # Create a group for the observed data
    obs_group = cell_group.create_group('observed')
    
    # Store the number of clusters as an attribute
    obs_group.attrs['n_clusters'] = cell_result['observed']['n_clusters']
    
    # Loop through the unique cluster labels
    for l in cell_result['observed']['points'].keys():
        if l == -1:  # Skip noise cluster
            continue
        
        # Create a subgroup for the cluster
        cluster_group = obs_group.create_group(str(l))
        # Store the number of spots in the cluster as an attribute
        cluster_group.attrs['n_spots'] = cell_result['observed']['n_spots_per_cluster'][l]
        # Store the points (as a subgroup)
        points_group = cluster_group.create_group('points')
        points_group.create_dataset('xs', data=cell_result['observed']['points'][l]['xs'])
        points_group.create_dataset('ys', data=cell_result['observed']['points'][l]['ys'])
        points_group.create_dataset('zs', data=cell_result['observed']['points'][l]['zs'])
        points_group.create_dataset('chroms', data=cell_result['observed']['points'][l]['chroms'].astype('S'))
        points_group.create_dataset('starts', data=cell_result['observed']['points'][l]['starts'])
        points_group.create_dataset('ends', data=cell_result['observed']['points'][l]['ends'])
        # Store the center of mass
        cluster_group.create_dataset('center', data=cell_result['observed']['center'][l])
        # Store the RMSD
        cluster_group.create_dataset('rmsd', data=cell_result['observed']['rmsd'][l])
    
    # If there are no randomizations, return
    if len(cell_result['nulls']) == 0:
        return cluster_h5
    
    # Otherwise, create a group for the null results
    nulls_group = cell_group.create_group('nulls')
    # Add the randomization results for each randomization
    for i in cell_result['nulls']:
        rnd_group = nulls_group.create_group(str(i))
        rnd_group.attrs['n_clusters'] = cell_result['nulls'][i]['n_clusters']
        # Store the number of spots per cluster as attributes
        for l in cell_result['nulls'][i]['n_spots_per_cluster'].keys():
            rnd_group.attrs[f'n_spots_cluster_{l}'] = cell_result['nulls'][i]['n_spots_per_cluster'][l]
    
    # Return the updated h5py file
    return cluster_h5


# MAIN FUNCTION TO RUN THE CLUSTERING

# Define the required keys for the configuration dictionary
required_keys = {
    'spots_selection_method': {'type': str},
    'eps': {'type': float, 'positive': True},
    'min_samples': {'type': int, 'positive': True},
    'h5_file': {'type': str},
    'n_randomizations': {'type': int, 'positive': True},
}

def clustering(
    cte: ChromatinTracingExperiment, scf: SingleCellFeature, config: dict
) -> h5py.File:
    """ Performs clustering of selected spots of interest in each cell of a
    Chromatin Tracing Experiment.
    
    The clustering is performed using the DBSCAN algorithm.
    
    The results are stored in an HDF5 file with the following structure:
    - cellID/
    -   observed/
    -      n_clusters (attribute)
    -      cluster_label/ (group)
    -         n_spots (attribute)
    -         points/ (group)
    -            xs (dataset)
    -            ys (dataset)
    -            zs (dataset)
    -            chroms (dataset)
    -            starts (dataset)
    -            ends (dataset)
    -         center (dataset)
    -         rmsd (dataset)
    -   nulls/
    -      randomization_index/ (group)
    -         n_clusters (attribute)
    -         n_spots_cluster_label (attribute for each cluster)
    
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
            - spots_selection_method (str): Method to select spots of interest.
            - n_randomizations (int): Number of randomizations to perform for null model.

    Returns:
        h5py.File: An HDF5 file containing clustering results for each cell.
    """
    
    cluster_h5 = parallel.control_func(
        cte, scf, config, required_keys,
        node_function, reduce_initialization, reduce_update
    )
    
    return cluster_h5