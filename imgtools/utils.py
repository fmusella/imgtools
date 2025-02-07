import os
import numpy as np
from scipy.spatial import distance
from scipy.stats import pearsonr
from scipy.ndimage import binary_dilation
import alphashape
import trimesh
import mrcfile
from alabtools.utils import Index


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


def get_alpha_mesh(alpha: float, points: np.ndarray) -> trimesh.Trimesh:
    """ Creates an alpha-shape mesh from a set of points. Depending on alpha:
        - If alpha is negative, raises an error.
        - If alpha is 0, fits a Convex Hull.
        - If alpha is positive, fits an alpha-shape.

    Args:
        alpha (float)
        points (np.ndarray): array of shape (npoints, 3) containing the 3D coordinates of the spots.

    Returns:
        trimesh.Trimesh: mesh fitted to the input points.
    """
    
    # If alpha is negative, raise an error
    if alpha < 0:
        raise ValueError("The alpha value must be positive.")
    
    # If alpha is 0, we fit a Convex Hull
    if alpha == 0:
        hull = trimesh.convex.convex_hull(points)
        return hull

    # Otherwise, we fit an alpha-shape
    shape = alphashape.alphashape(points, alpha)
    mesh = trimesh.Trimesh(vertices=shape.vertices, faces=shape.faces, process=True)
    
    return mesh

def fit_alphashape(points: np.ndarray, alpha: float, force: bool, reducing_factor: float = 0.5) -> (float, trimesh.Trimesh):
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
        reducing_factor (int, optional): factor by which the alpha value is multiplied at each iteration. Defaults to 0.5
    
    Returns:
        alpha_ (float): output alpha value, could be different from the input one if force=False.
        mesh (trimesh.Trimesh): alpha-shape fitted to the input points.
    """
    
    # The alphashape code doesn't give closed shapes if the input points are not float64
    points = points.astype(np.float64)
    
    # If force or alpha is 0, we try to fit the alpha-shape with the input alpha value,
    # and if the shape is not closed, we raise an error.
    if force or alpha == 0:
        mesh = get_alpha_mesh(alpha, points)
        if not mesh.is_watertight:
            raise ValueError("The alpha-shape is not closed with the input alpha value forced. Try setting force=False.")
        return alpha, mesh
    
    # Otherwise, we find the alpha value by a search algorithm,
    # where we start with the input alpha and - if the shape is not closed - we decrease it.
    max_iter = 10  # maximum number of iterations
    counter = 0
    alpha_ = alpha  # new alpha value, to be iteratively decreased
    while True:
        counter += 1
        if counter > max_iter:
            raise ValueError("Maximum number of iterations reached, but no closed alpha-shape found.")
        mesh = get_alpha_mesh(alpha_, points)
        if mesh.is_watertight:
            return alpha_, mesh
        alpha_ = alpha_ * reducing_factor


def write_cmm(
    filename: str, marker_str: str, coord: np.ndarray, radius: float,
    color: np.ndarray = [0, 0, 0], links: np.ndarray = None
) -> None:
    """ Write a CMM file.
    
    Only works for a single marker set. Colors all markers and links with the same color.

    Args:
        filename (str): name of the file to be written
        marker_str (str): string to identify the marker set
        coord (np.ndarray): numpy array of shape (n_markers, 3)
                containing the coordinates of the markers
        radius (float): size of the markers (in physical units)
        color (np.ndarray, optional): numpy array of shape with the colors of markers and links.
                Can be either (3,) or (n_markers, 3). Defaults to [0, 0, 0]
        links (np.ndarray, optional): numpy array of shape (n-1,),
                True if there is a link between i and i+1. Defaults to None (no links)
    """

    with open(filename,'w') as f:
        
        if color.shape == (3,):
            color = np.tile(color, (len(coord), 1))
        
        f.write('<marker_set name="marker set %s">\n' % marker_str)
        
        # Write markers
        for i in range(len(coord)):
            f.write(
                '<marker id="%d" x="%.3f" y="%.3f" z="%.3f" r="%.3f" g="%.3f" b="%.3f" radius="%.3f" note="" nr="%.3f" ng="%.3f" nb="%.3f"/>\n'
                    % (i + 1, coord[i, 0], coord[i, 1], coord[i, 2],
                       color[i, 0], color[i, 1], color[i, 2],
                       radius, color[i, 0], color[i, 1], color[i, 2])
            )
        
        if links is None:
            f.write('</marker_set>\n')
            return None
        
        # Write links
        for i in range(len(coord) - 1):
            # Skip if there is no link between i and i+1
            if not links[i]:
                continue
            # Otherwise, write the link
            f.write(
                '<link id1="%d" id2="%d" r="%.3f" g="%.3f" b="%.3f" radius="%.3f" />\n'
                    % (i + 1, i + 2, color[i, 0], color[i, 1], color[i, 2], radius / 4)
            )
        
        f.write('</marker_set>\n')


# MRC FUNCTIONS

def mesh_to_mrc(
    path: str,
    name_prefix: str,
    mesh: trimesh.Trimesh,
    resolution: float,
    border: int,
    ndilation: int = None
) -> tuple:
    """ Save a mesh as a MRC file.

    Args:
        path (str): directory where the MRC file will be saved
        name_prefix (str): prefix of the MRC file name
        mesh (trimesh.Trimesh): mesh used to create the MRC file
        resolution (float): voxel size of the MRC file (in physical units)
        border (int): black border around the mesh (in voxels)
        ndilation (int, optional): number of dilations to apply to the mask. Defaults to None.

    Returns:
        origin_mrc_vx (tuple): origin of the MRC file in voxel units
        shape (tuple): shape of the MRC file
    """
    
    # Get the bounding box of the mesh
    bbox = mesh.bounding_box.bounds  # np.array of shape (2, 3)
    
    # Quantize the bounding box by the resolution
    bbox = resolution * np.round(bbox / resolution)
    
    # Add the border (multiplied by the resolution) to the bounding box
    bbox[0] -= border * resolution
    bbox[1] += border * resolution
    
    # Create 3D grid
    xyz, shape = create_grid(bbox, resolution)
    
    # Use mesh.contains() to create a boolean 3D mask of the volume
    volume_mask = mesh.contains(xyz).reshape(shape).astype(int)
    
    # Dilate the mask to add more depth to the mesh
    if ndilation is not None:
        volume_mask = binary_dilation(volume_mask, iterations=ndilation)
    
    # Get the origin of the mrc file in voxel units, so that it matches with the imaging spots
    # It is the first point of the bounding box, quantized by the resolution
    origin_mrc_vx = np.round(bbox[0] / resolution).astype(int)
    
    # Save the volume mask as a MRC file
    write_mrc(
        filename = os.path.join(path, name_prefix + '.mrc'),
        data = volume_mask,
        origin = tuple(origin_mrc_vx),
        voxel_size = (resolution, resolution, resolution)
    )
    
    return origin_mrc_vx, shape

def write_mrc(
    filename: str,
    data: np.ndarray,
    origin: tuple = (0, 0, 0),
    voxel_size: tuple = (1, 1, 1)
) -> None:
    """Write a MRC file from a numpy array.

    Args:
        filename (str): name of the file to be written.
        data (np.array(shape=(n_x_grid, n_y_grid, n_z_grid))): grid of values (0 or 1)
        origin (tuple, optional): origin of the MRC file in voxel units. Defaults to (0, 0, 0).
        voxel_size (tuple, optional): voxel size of the MRC file in physical units. Defaults to (1, 1, 1).
    """
    
    # Check that the parent directory exists
    if not os.path.exists(os.path.dirname(filename)):
        raise ValueError('The parent directory does not exist')
    
    # Swap the axes to match the MRC format
    data = np.swapaxes(data, 0, 2)
    # If the data is boolean or integer, convert it to int8
    if data.dtype == bool or np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.int8)
    # If the data is float, convert it to float32
    elif np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)
    else:
        raise ValueError('The data type is not supported')
    # Create a new MRC file and save the data
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.nstart = origin
        mrc.voxel_size = voxel_size

def create_grid(bbox: np.array, resolution: float) -> tuple:
    """ Create a 3D grid of points.

    Args:
        bbox (np.array):
            array of shape (2, 3) containing the min and max values of the bounding box
        resolution (float): resolution of the grid
    
    Returns:
        xyz (np.array): array of shape (n_points, 3) containing the coordinates of the points
        shape (tuple): shape of the grid (n_x_grid, n_y_grid, n_z_grid)
    """
    xs = np.arange(bbox[0, 0], bbox[1, 0], resolution)
    ys = np.arange(bbox[0, 1], bbox[1, 1], resolution)
    zs = np.arange(bbox[0, 2], bbox[1, 2], resolution)
    xyz = list()
    for x in xs:
        for y in ys:
            for z in zs:
                xyz.append(np.array([x, y, z]))
    xyz = np.array(xyz)
    shape = (len(xs), len(ys), len(zs))
    return xyz, shape


def compare_index(idx1: Index, idx2: Index, usechr: list) -> bool:
    """Compares two Index objects on a subset of chromosomes.

    Args:
        idx1 (Index): first Index object.
        idx2 (Index): second Index object.
        usechr (list): list of chromosomes to be used in the comparison.

    Returns:
        bool: True if the two Index objects are the same on the subset of chromosomes.
    """
    
    if idx1.genome.assembly != idx2.genome.assembly:
        return False
    
    # Compare the two Index objects on the chromosomes in usechr
    if np.any(idx1.chromstr[np.isin(idx1.chromstr, usechr)] != idx2.chromstr[np.isin(idx2.chromstr, usechr)]):
        return False
    if np.any(idx1.start[np.isin(idx1.chromstr, usechr)] != idx2.start[np.isin(idx2.chromstr, usechr)]):
        return False
    if np.any(idx1.end[np.isin(idx1.chromstr, usechr)] != idx2.end[np.isin(idx2.chromstr, usechr)]):
        return False
    
    return True


def smooth(x: np.array, chromstr: np.array, k: int) -> np.array:
    """ Smooth a signal by chromosome.
    Uses the convolution of the signal with a uniform filter of size k,
    with the function np.convolve.

    Args:
        x (np.array): array to smooth
        chromstr (np.array): chromosome array of x
        k (int): window size of the smoothing kernel

    Returns:
        x_smooth (np.array): smoothed array
    """
    
    # Initialize the smoothed array
    x_smooth = np.copy(x)
    
    # Loop over chromosomes and smooth the signal
    for chrom in np.unique(chromstr):
        mask = chromstr == chrom
        # Define the kernel, which is a uniform filter of size k
        kernel = np.ones(k) / k
        x_smooth[mask] = np.convolve(x[mask], kernel, mode='same')
    
    return x_smooth


def clean_pearsonr(x: np.array, y: np.array) -> float:
    """Pearson correlation coefficient, ignoring NaNs and Infs.

    Args:
        x (np.array(n), dtype=float): first input array.
        y (np.array(n), dtype=float): second input array.
    
    Returns:
        (float): Pearson correlation coefficient.
    """
    
    # Convert Infs to NaNs
    x[np.isinf(x)] = np.nan
    y[np.isinf(y)] = np.nan
    
    # Remove NaNs (from both arrays)
    mask = np.logical_and(~np.isnan(x), ~np.isnan(y))
    x = x[mask]
    y = y[mask]
    
    # Compute Pearson correlation coefficient
    return pearsonr(x, y)[0]


def convert_to_abs_path(cfg: dict):
  """ Given a dictionary or arbitrary depth, convert all relative paths to absolute paths.
  
  This is a recursive function: for each key-value pair, if the value is a file path, it is converted to an absolute path.
  Otherwise, if the value is a dictionary, the function is called recursively on the value.

  Args:
    cfg (dict): Dictionary of arbitrary depth.
  """
  for key, value in cfg.items():
    # If the value is a dictionary, call the function recursively
    if isinstance(value, dict):
      convert_to_abs_path(value)
    # If the value is a file path, convert it to an absolute path
    elif isinstance(value, str) and os.path.exists(value):
      cfg[key] = os.path.abspath(value)


def resample_array(n: int, arr: np.ndarray) -> np.ndarray:
    """ Given an input array, generates a resampled and shuffled
    array of size n.
    
    n can either be equal, less than or more than the length of the input array:
        - if n is equal to the length of arr, the output array is a shuffled version of arr,
        - if n is less than the length of arr, the output array is sampled from the array
            without replacement,
        - if n is more than the length of arr, the output array is created by first tiling
            the input array as much as possible, and then randomly selecting the rest without
            replacement. This ensures that each different element is used as much as possible.

    Args:
        n (int): Size of the resampled array.
        arr (np.ndarray): The array to sample from.

    Returns:
        np.ndarray: Resampled and shuffled array of size n.
    """
    
    # Calculate the number of repetitions and the remainder: n = a * len(idx) + b
    # For example, if n = 430 and len(arr) = 200, then a = 2 and b = 30
    a = n // len(arr)
    b = n % len(arr)

    # Create the repeated part and the remainder part
    arr_out_a = np.tile(arr, a)
    arr_out_b = np.random.choice(arr, b, replace=False)

    # Concatenate the repeated and remainder parts
    arr_out = np.concatenate([arr_out_a, arr_out_b])

    # Shuffle the final array to ensure randomness
    np.random.shuffle(arr_out)

    return arr_out
