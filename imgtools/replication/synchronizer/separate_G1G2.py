import numpy as np
from sklearn.cluster import AgglomerativeClustering
from ...scf import SingleCellFeature


def separate_G1G2(scf: SingleCellFeature, states: np.ndarray) -> np.ndarray:
    """ Separate G1 and G2 cells based on the number of spots and the cell volume.
    
    The function clusters the G cells into 2 groups based on the cell volume and total number of imaged spots.
    
    The clusters are identified with the AgglomerativeClustering algorithm.
    Then, the cluster with the smaller volume and total number of spots is called 'G1' and the other 'G2'.
    If the clusters are not well separated, the function raises an error.

    Args:
        scf (SingleCellFeature)
        states (np.ndarray): Array of cell states with 'S' for S cells and 'G' for either G1 or G2 cells

    Returns:
        states_new (np.ndarray): Array of cell states with 'S' for S cells and 'G1'/'G2' for G1/G2 cells
    """

    # Get the cell nuclei volumes
    volumes = scf.volumes
    # Get the spot-count matrix and the total number of spots for each cell
    spotcount = scf.get_feature('spotcount')  # (ncell, ndomain, ncopy)
    totspots = np.sum(spotcount, axis=(1, 2))  # (ncell,)
    
    # Check that the states array has the right shape
    if len(states) != len(volumes):
        raise ValueError('Dimension mismatch between states and volumes')

    # Cluster the G cells into 2 groups based on totspots and volumes
    X = np.column_stack((volumes[states == 'G'], totspots[states == 'G']))
    agg = AgglomerativeClustering(n_clusters=2).fit(X)
    labels = agg.labels_
    
    # Get the centers of the clusters
    center_0 = np.mean(X[labels == 0], axis=0)  # (2,)
    center_1 = np.mean(X[labels == 1], axis=0)  # (2,)
    
    # Call 'G1' the cluster with the smaller volume/totspots
    states_G = np.full(len(labels), '').astype(states.dtype)
    if center_0[0] < center_1[0] and center_0[1] < center_1[1]:
        states_G[labels == 0] = 'G1'
        states_G[labels == 1] = 'G2'
    elif center_0[0] > center_1[0] and center_0[1] > center_1[1]:
        states_G[labels == 0] = 'G2'
        states_G[labels == 1] = 'G1'
    else:
        raise ValueError('Cannot determine G1/G2: the clusters are not well separated')

    # Create a new states array that is 'S' for S cells and 'G1'/'G2' for G cells
    states_new = np.full(len(states), '').astype(states.dtype)
    idx_G = np.where(states == 'G')[0]
    assert len(idx_G) == len(states_G)
    idx_G1 = idx_G[states_G == 'G1']
    idx_G2 = idx_G[states_G == 'G2']
    states_new[idx_G1] = 'G1'
    states_new[idx_G2] = 'G2'
    states_new[states == 'S'] = 'S'
    
    return states_new
