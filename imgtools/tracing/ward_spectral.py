import numpy as np
from scipy.spatial import distance
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering

class WardSpectralClustering:
    """
    It implements a combination of Ward and Spectral clustering,
    as described in Takei et al. Nature (2021). 
    Reference: https://www.nature.com/articles/s41586-020-03126-2
    """
    
    def __init__(
        self,
        n_clusters: int = 2,
        st: float = 1.2,
        ot: float = 30.
    ):  
        self.n_clusters = n_clusters
        self.st = st
        self.ot = ot

    def fit(self, X: np.ndarray):
        
        # First, we try to perform Ward clustering
        labels = AgglomerativeClustering(n_clusters=self.n_clusters, linkage='ward').fit(X).labels_
        
        # If the clusters are not separated enough, we need to perform spectral clustering
        if not are_separated(X, labels, self.st):        
            # Compute the distance matrix
            dist_mat = distance.cdist(X, X, 'euclidean')
            # The affinity matrix is needed for the Spectral Clustering
            # This transformation is advised by sklearn (https://scikit-learn.org/dev/modules/clustering.html)
            beta = 1  # This parameter can be tuned, but 1 seems to give good results
            aff_mat = np.exp(- beta * dist_mat / dist_mat.std())
            labels = SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed').fit(aff_mat).labels_
        
        # Remove outliers
        labels = remove_outliers(X, labels, self.ot)
        
        self.labels_ = labels
    

def compute_centroids(X: np.ndarray, labels: np.ndarray):
    """Computes centroids and spreads of a set of points with their clustering labels.

    Args:
        pts (np.array(n,3)): 3D coordinates of points to cluster
        labels (np.array(n,)): labels of the clusters

    Returns:
        ctr: dict of np.array(3,): list of centroids of each cluster
        spd: dict of np.array(3,): list of spreads of each cluster
    """
    
    ctr, spd = {}, {}  # dict of centroid and spreads for each cluster
    
    for l in np.unique(labels):
        if l == -1:
            continue
        X_l = X[labels == l]
        ctr[l] = np.nanmean(X_l, axis=0)  # Compute the centroids
        spd[l] = np.nanstd(X_l, axis=0)  # Compute the standard deviation of the points
    
    return ctr, spd
    
def are_separated(X: np.ndarray, labels: np.ndarray, st: float):
    """Checks if the two clusters are separated enough.

    Args:
        pts (np.array(n,3)): 3D coordinates of points to cluster
        labels (np.array(n,)): labels of the clusters
        st (float): separation threshold

    Returns:
        bool: True if the clusters are separated enough, False otherwise
    """
    
    if len(np.unique(labels)) <= 1:
        return True
    
    ctr, spd = compute_centroids(X, labels)
    for i, l1 in enumerate(ctr.keys()):
        for j, l2 in enumerate(ctr.keys()):
            if j <= i:
                continue
            ctr1, spd1 = ctr[l1], spd[l1]
            ctr2, spd2 = ctr[l2], spd[l2]
            # Separation criterion:
            # the distance between the centroids is smaller than the sum of the spreads
            # multiplied by the factor st
            if np.linalg.norm(ctr1 - ctr2) <= st * (np.linalg.norm(spd1) + np.linalg.norm(spd2)):
                return False
    
    return True

def remove_outliers(X: np.ndarray, labels: np.ndarray, ot: float):
        """Removes outliers from the set of points.
        Outliers are identified as points that are too far from the centroid of their cluster.

        Args:
            pts (np.array(n,3)): 3D coordinates of points to cluster
            lbl (np.array(n,)): labels of the clusters
            ot (float): outlier threshold

        Returns:
            lbl_crrect (np.array(n,)): labels of the clusters with outliers removed (i.e. set to -1)
        """
        
        ctr, spd = compute_centroids(X, labels)
        labels_correct = labels.copy()
        
        for i in range(len(labels)):
            p = X[i]
            l = labels[i]
            if np.linalg.norm(p - ctr[l]) > ot * np.linalg.norm(spd[l]):
                labels_correct[i] = -1
        
        return labels_correct
