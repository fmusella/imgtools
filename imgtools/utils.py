import numpy as np
from scipy.spatial import distance
import alphashape
import trimesh

def spots_3d_median(points: np.ndarray, centroid: np.ndarray) -> int:
    """ Given a list of spot points associated with the same domain in a trace,
    selects only one of them with the 3D median criterion.
    
    There are three cases:
        1) If there is only one point, return the index of that point (0)
        2) If there are two points, return the one closer to the centroid
        3) If there are three or more points, return the point that minimizes the sum of distances to all other points
    
    The function returns the index of the selected point.
    
    The computation for point 3, the actual 3D median, is based on the 3D geometric median (https://en.wikipedia.org/wiki/Geometric_median).
    However, in this reference the algorithm finds the point in 3D space that minimizes the sum of distances to all other points,
    so it doesn't return a point of the set. Here we don't want to create 'fake' points, so I adapted the algorithm to only
    consider the points in the set.
    
    Args:
        points (np.ndarray): array of shape (npoints, 3) containing the 3D coordinates of the spots.
        centroid (np.ndarray): array of shape (3,) containing the 3D coordinates of the centroid of the trace.

    Returns:
        median_idx (int): index of the selected point, between 0 and npoints-1.
    """
    
    # Check the points array
    if not isinstance(points, np.ndarray):
        raise TypeError('points must be a numpy array')
    if len(points.shape) != 2:
        raise ValueError('points must be a numpy array of shape (npoints, 3)')
    if points.shape[1] != 3:
        raise ValueError('points must be a numpy array of shape (npoints, 3)')
    if points.shape[0] == 0:
        raise ValueError('points must be a numpy array of shape (npoints, 3)')
    npoints = points.shape[0]  # get the number of points
    
    # Check the centroid array
    if not isinstance(centroid, np.ndarray):
        raise TypeError('centroid must be a numpy array')
    if centroid.shape != (3,):
        raise ValueError('centroid must be a numpy array of shape (3,)')
    
    # If there is only one point, return the index of that point (0)
    if npoints == 1:
        return 0
    
    # If there are two points, return the one closer to the centroid
    elif npoints == 2:
        median_idx = 0 if distance.euclidean(points[0], centroid) <= distance.euclidean(points[1], centroid) else 1
        return median_idx
    
    # Otherwise, find the point that minimizes the sum of distances to all other points
    # Initialize the median index and the minimum distance
    median_idx = None
    dists_min = np.inf
    # Loop over the points
    for i in range(len(points)):
        dists_i = 0  # initialize the sum of distances for point i
        for j in range(len(points)):
            if i == j:
                continue
            dists_i = dists_i + distance.euclidean(points[i], points[j])  # add the distance between point i and point j
        # If the total distance is smaller than the minimum distance, update the median index and the minimum distance
        if dists_i < dists_min:
            median_idx = i
            dists_min = dists_i
        # If the total distances are equal, choose the point closer to the centroid
        if dists_i == dists_min:
            if distance.euclidean(points[i], centroid) < distance.euclidean(points[median_idx], centroid):
                median_idx = i
                dists_min = dists_i
    return median_idx


def fit_alphashape(points: np.ndarray, alpha: float, force: bool) -> (float, trimesh.Trimesh):
    """
    Fits an alpha-shape to contain all the points in the cell.
    
    If force is True, the alpha-shape is fitted with the input alpha value.
    
    Otherwise, the alpha value is found by a search algorithm starting from the input one
    and halving it until a closed alpha-shape is found.
    A hard-coded maximum number of iterations is used to avoid infinite loops.
    
    Args:
        points (np.ndarray): array of shape (npoints, 3) containing the 3D coordinates of the spots.
        alpha (float): input alpha value.
        force (bool): if True, the alpha value is not changed.
    
    Returns:
        alpha_ (float): output alpha value, could be different from the input one if force=False.
        mesh (trimesh.Trimesh): alpha-shape fitted to the input points.
    """
    
    # The alphashape code doesn't give closed shapes if the input points are not float64
    points = points.astype(np.float64)
    
    # If force, we only use the input alpha value
    if force:
        alpha_shape = alphashape.alphashape(points, alpha)
        mesh = trimesh.Trimesh(vertices=alpha_shape.vertices, faces=alpha_shape.faces, process=True)
        if not mesh.is_watertight:
            raise ValueError("The alpha-shape is not closed with the input alpha value forced. Try setting force=False.")
        return alpha, mesh
    
    # If not force, we find the alpha value by a search algorithm,
    # where we start with the input alpha and - if the shape is not closed - we halve it.
    max_iter = 20
    counter = 0
    alpha_ = alpha  # new alpha value, to be iteratively halved
    while True:
        counter += 1
        if counter > max_iter:
            raise ValueError("Maximum number of iterations reached, but no closed alpha-shape found.")
        alpha_shape = alphashape.alphashape(points, alpha_)
        mesh = trimesh.Trimesh(vertices=alpha_shape.vertices, faces=alpha_shape.faces, process=True)
        if mesh.is_watertight:
            return alpha_, mesh
        alpha_ = alpha_ / 2
