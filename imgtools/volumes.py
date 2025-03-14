import os
import h5py
import numpy as np
from scipy.stats import gaussian_kde
from scipy.ndimage import binary_dilation, binary_erosion, label
import trimesh
from .cte import ChromatinTracingExperiment
from .scf import SingleCellFeature
from . import parallel


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


# SINGLE CELL BODIES' VOLUMES KERNEL DENSITY ESTIMATION

def run_bodies_single_cell(
    cellID: str, cte: ChromatinTracingExperiment, scf: SingleCellFeature,
    voxel_res: float, kde_alpha: float, bodies_to_features: dict,
    border: int = 10,
):
    """ 
    Run the nuclear bodies detection algorithm for a single cell.
    
    Does the following:
    - Creates a KDE density for the cell spots, inverting it so that the "negative density" is obtained.
    - For each nuclear body, creates a KDE using only spots with a high body features value.
    - For each nuclear body, combines the previous two KDEs: uses the "missing volume" KDE, but only
      when the KDE of the body is higher than the KDE of the other bodies.
    
    Args:
        cellID (str)
        cte (ChromatinTracingExperiment)
        scf (SingleCellFeature)
        voxel_res (float): length of the (cubic) voxel edge (in same units as the coordinates in CTE)
        kde_alpha (float): alpha parameter for the Kernel Density Estimation
        bodies_to_features (dict): dictionary with the bodies as keys and the features as values. E.g.
                bodies_to_features = {
                    'Nucleoli': ['Fibrillarin', 'rDNA', 'Rnu3b_RNA', 'ITS1_RNA'],
                    'Centromeres-Telomeres': ['MajSat', 'MinSat', 'Telomere'],
                    'Speckles': ['SF3A66'],
                }
        border (int, optional): white border around the cell nucleus. Defaults to 10.

    Returns:
        images (dict[str: np.array]): dictionary where the keys are the bodies' names and the values are the
            3D images of the bodies (np.array of shape shape). These images have continuous values.
        origin_voxel (np.array): origin of the MRC in voxel units
        bbox (np.array): bounding box of the MRC
        shape (tuple): shape of the MRC
    """
    
    # Get the mesh of the cell
    try:
        mesh = cte.get_alphashapes(cellID)['mesh']
    except Exception as e:
        raise Exception(f'Error getting the mesh of cell {cellID}: {e}')
    
    # Get the bounding box of the mesh
    bbox = mesh.bounding_box.bounds  # np.array of shape (2, 3)
    # Quantize the bounding box by the resolution
    bbox = voxel_res * np.round(bbox / voxel_res)
    # Add the border (multiplied by the resolution) to the bounding box
    bbox[0] -= border * voxel_res
    bbox[1] += border * voxel_res
    # Calculate the origin of the MRC in voxel units
    origin_voxel = np.round(bbox[0] / voxel_res).astype(int)
    
    # Get the coordinates of the spots of the cell
    xs, ys, zs, _, _, _, _, _, _ = cte.get_data(cellID, format='numpy')
    crd = np.array([xs, ys, zs]).T
    
    # Calculate the Gaussian Kernel Density Estimate
    kde = gaussian_kde(crd.T, bw_method=kde_alpha)
    
    # Create 3D grid
    XYZ, shape = create_grid(bbox, voxel_res)
    
    # Calculate the gaussian KDE on the 3D grid and invert the values,
    # to create the 'negative density' image
    kXYZ_negden = kde(XYZ.T).reshape(shape)
    kXYZ_negden = np.max(kXYZ_negden) - kXYZ_negden
    
    # Set as 0 voxels that are too close to the nuclear envelope,
    # since it's just confounding to determine nuclear bodies
    surface_dists = trimesh.proximity.signed_distance(mesh, XYZ).reshape(shape)
    threshold = 0.75  # 750 nm
    kXYZ_negden[surface_dists < threshold] = 0
    
    # Create new KDE densities from the spots with the highest feature values
    # (remember that to each body corresponds a set of features)
    kXYZ_bodies = {}
    for body in bodies_to_features:
        
        # Initialize the list of feature values, to be converted later to a numpy array
        # of shape (n_features, n_spots)
        bodyvals = []
        
        # Loop over the features of the body
        for feat in bodies_to_features[body]:
            
            # Get the feature values for the feature, shape (n_spots,)
            featvals = scf.get_feature_by_spotIDs(cellID, cte, feat).astype(float)
            bodyvals.append(featvals)
        
        # Convert the list to array
        bodyvals = np.array(bodyvals)  # shape (n_features, n_spots)
        # Take the maximum value among the features for each spot
        bodyvals = np.nanmax(bodyvals, axis=0)  # shape (n_spots,)
        
        # Select the body-associated spots as those with a body value above a X percentile
        percentile = 80
        threshold = np.nanpercentile(bodyvals, percentile)
        xs_body = xs[bodyvals > threshold]
        ys_body = ys[bodyvals > threshold]
        zs_body = zs[bodyvals > threshold]
        crd_body = np.array([xs_body, ys_body, zs_body]).T

        # Calculate the Gaussian Kernel Density Estimate from the selected body spots
        kde = gaussian_kde(crd_body.T, bw_method=kde_alpha)
        
        # Calculate the KDE on the 3D grid
        kXYZ_bodies[body] = kde(XYZ.T).reshape(shape)
    
    # Now create the images of the bodies
    images = {}
    for body in bodies_to_features:
        # Initialize the image of the body as a copy of the negative density image
        kXYZ_body = np.copy(kXYZ_negden)
        # Now loop over the other bodies and set the voxels of kXYZ_body to 0
        # whenever the KDE intensity from other bodies is higher
        for body2 in bodies_to_features:
            if body == body2:
                continue
            kXYZ_body[kXYZ_bodies[body] <= kXYZ_bodies[body2]] = 0
        # Save the image
        images[body] = kXYZ_body
    
    return images, origin_voxel, bbox, shape

# PARALLEL FUNCTIONS FOR THE BODIES' VOLUMES KERNEL DENSITY ESTIMATION

def run_bodies(cte: ChromatinTracingExperiment, scf: SingleCellFeature, config: dict) -> None:
    """ Parallel code to identify nuclear bodies in each cell and save them in an HDF5 file.
    
    The HDF5 file will have the following structure:
    - One group for each cell, with the cellID as the group name.
    - Each group will have the following datasets:
        - origin: origin of the MRC in voxel units
        - bbox: bounding box of the MRC
        - shape: shape of the MRC
        - One dataset for each body, with the body name as the dataset name and the 3D image as the dataset value.
    
    Importantly, these images are continuous. To obtain binary images, use the binarize function.
    
    The config specifies the parameters of the calculation:
    - h5_name: name of the output HDF5 file
    - voxel_resolution: resolution of the voxel edge (in the same units as the coordinates in CTE)
    - kde_alpha: alpha parameter for the Kernel Density Estimation
    - bodies_to_features: dictionary with the bodies as keys and the features as values. E.g.
        bodies_to_features = {
            'Nucleoli': ['Fibrillarin', 'rDNA', 'Rnu3b_RNA', 'ITS1_RNA'],
            'Centromeres-Telomeres': ['MajSat', 'MinSat', 'Telomere'],
            'Speckles': ['SF3A66'],
        }
    - border: white border around the cell nucleus

    Args:
        cte (ChromatinTracingExperiment)
        scf (SingleCellFeature)
        config (dict)
    """
    
    # Define the required keys for the config
    required_keys = {
        'h5_name': {'type': str},
        'voxel_resolution': {'type': float, 'positive': True},
        'kde_alpha': {'type': float, 'positive': True},
        'bodies_to_features': {'type': dict},
        'border': {'type': int, 'positive': True},
    }
    
    def _rfunc_init(_1, _2, _3, _4) -> dict:
        """ Initialization function for reduction step.
        
        Args:
            _*: not used, just to match the signature of the function
        
        Returns:
            dict: empty dictionary
        """
        return {}

    def _rfunc_update(cellID: str, bodies: dict, cell_bodies: dict, _1, _2, _3) -> dict:
        """ Update function for reduction step.
        
        bodies is a dictionary with structure:
           bodies[cellID] = cell_bodies
        cell_bodies is a dictionary with structure:
           cell_bodies = {
               'images': dict[str: np.array],
               'origin': np.array,
               'bbox': np.array,
               'shape': tuple,
           }

        Args:
            cellID (str)
            bodies (dict): dictionary with the bodies of all cells
            cell_bodies (dict): dictionary with the bodies of the current cell
            _*: not used, just to match the signature of the function

        Returns:
            dict: _description_
        """
        bodies[cellID] = cell_bodies
        return bodies
    
    def _nfunc(cellID: str, cte_name: str, scf_name: str, config: dict) -> dict:
        """ Node function to calculate the bodies of a single cell.
        
        Just a wrapper around run_bodies_single_cell.

        Args:
            cellID (str)
            cte_name (str)
            scf_name (str)
            config (dict): configuration dictionary with the parameters as defined in required_keys

        Returns:
            cell_bodies (dict): dictionary with the bodies of the cell
        """
        
        # Open the CTE and SCF
        cte = ChromatinTracingExperiment(cte_name, 'r')
        scf = SingleCellFeature(scf_name, 'r')
        
        # Unpack the parameters from config
        voxel_res = config['voxel_resolution']
        kde_alpha = config['kde_alpha']
        bodies_to_features = config['bodies_to_features']
        border = config['border']
        
        # Run the bodies calculation for the cell
        images, origin, bbox, shape =  run_bodies_single_cell(
            cellID, cte, scf, voxel_res, kde_alpha, bodies_to_features, border
        )
        
        # Return a dictionary with the results
        cell_bodies = {'images': images, 'origin': origin, 'bbox': bbox, 'shape': shape} 
        return cell_bodies
    
    # Run the bodies calculation in parallel
    bodies = parallel.control_func(
        cte,
        scf,
        config,
        required_keys,
        _nfunc,
        _rfunc_init,
        _rfunc_update
    )
    
    # To save the results, first convert the filename to its absolute path
    h5_name = config['h5_name']
    h5_name = os.path.abspath(h5_name)
    # Make sure the directory exists
    os.makedirs(os.path.dirname(h5_name), exist_ok=True)
    
    # Create the HDF5 file
    with h5py.File(h5_name, 'w') as f:
        # Loop over cells
        for cellID, cell_bodies in bodies.items():
            # Create a group for the cell
            cell_group = f.create_group(cellID)
            # Save origin, bbox, shape in the group (they are the same for all bodies of the cell)
            cell_group.create_dataset('origin', data=cell_bodies['origin'])
            cell_group.create_dataset('bbox', data=cell_bodies['bbox'])
            cell_group.create_dataset('shape', data=cell_bodies['shape'])
            # Save each body in the group
            for body, image in cell_bodies['images'].items():
                cell_group.create_dataset(body, data=image)


# BINARIZATION

def binarize(
    image: np.ndarray, percentile: float, voxel_res: float, nerosion: int = 1, ndilation: int = 1
):
    """ Binarize the continuous-valued KDE image of a nuclear body in a single cell.
    
    Voxels above a certain percentile of the KDE non-zero values are set to 1,
    and the rest are set to 0.
    
    The percentile is found by using a bisection method search, where the goal
    is to obtain a ratio between the total volume of the body and the volume of
    the nucleus that is close to a target ratio.
    
    Erosion and dilation can be applied to remove small objects and fill holes.

    Args:
        image (np.ndarray): continuous-valued KDE image of the body
        cell_volume (float): volume of the cell nucleus
        target_ratio (float): target ratio between
            the total volume of the body and the nucleus volume
        voxel_res (float): length of the (cubic) voxel edge
            (in same units as the coordinates in CTE)
        nerosion (int, optional): Number of iterations for the erosion. Defaults to 1.
            If 0, no erosion is performed.
        ndilation (int, optional): Number of iterations for the dilation. Defaults to 1.
            If 0, no dilation is performed.

    Returns:
        bimage (np.ndarray): binarized image of the body
        percentile (float): percentile used to binarize the image
            (optimized by the bisection method)
        ratio (float): ratio between the total volume of the body and the nucleus volume
            (optimized by the bisection method)
    """
    
    # Binarize the image using the current percentile
    threshold = np.nanpercentile(image[image > 0], percentile)
    bimage = (image >= threshold).astype(int)
    
    # Perform erosion and dilation to remove small objects and fill holes
    if nerosion > 0:
        bimage = binary_erosion(bimage, iterations=nerosion)
    if ndilation > 0:
        bimage = binary_dilation(bimage, iterations=ndilation)
    
    # Identify connected components
    limage, nbodies = label(bimage)
    
    return bimage, limage, nbodies
