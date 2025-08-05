import os
import h5py
import numpy as np
from scipy.stats import gaussian_kde
from scipy.ndimage import binary_dilation, binary_erosion, label
import trimesh
import alphashape
from .cte import ChromatinTracingExperiment
from .scf import SingleCellFeature
from . import parallel


# ALPHASHAPE / CONVEXHULL CELL NUCLEI VOLUMES' FITTING

def run_alphashape_single_cell(
    cellID: str, cte: ChromatinTracingExperiment, alpha: float, force: bool
) -> dict:
    """ Fit an alpha-shape to a single cell.

    Args:
        cellID (str)
        cte (ChromatinTracingExperiment)
        alpha (float): alpha value for the alphashape
        force (bool): if True, the alpha value is not changed

    Returns:
        cell_alphamesh (dict): dictionary with the alpha value and the mesh of the cell, as follows:
            {'alpha': float, 'mesh': trimesh.Trimesh}
    """
    
    # Get the data of the cell
    d = cte.get_data(cellID, format='numpy')
    xs, ys, zs = d['xs'], d['ys'], d['zs']
    points = np.array([xs, ys, zs]).T
    
    # Fit the alphashape
    alpha, mesh = fit_alphashape(points, alpha, force)
    
    # Return the alpha value and the mesh
    cell_alphamesh = {'alpha': alpha, 'mesh': mesh}
    
    return cell_alphamesh

def run_alphashape(cte: ChromatinTracingExperiment, config: dict) -> None:
    """ Parallel code to fit an alpha-shape to each cell.
    
    Required keys in the config:
    - alpha (float, positive): alpha value for the alphashape
    - force (bool): if True, the alpha value is not changed

    If alpha = 0, the Convex Hull is fitted.
    
    Args:
        cte (ChromatinTracingExperiment)
        config (dict)
    """
    
    # Define the required keys for the config
    required_keys = {
        'alpha': {'type': float, 'positive': True},
        'force': {'type': bool},
    }
    
    def _rfunc_init(_1, _2, _3, _4) -> dict:
        """ Initialization function for reduction step.
        
        Args:
            _*: not used, just to match the signature of the function
        
        Returns:
            dict: empty dictionary
        """
        return {}

    def _rfunc_update(cellID: str, alphameshes: dict, cell_alphamesh: dict, _1, _2, _3) -> dict:
        """ Update function for reduction step.
        
        alphameshes is a dictionary with structure:
           alphameshes[cellID] = cell_alphamesh
        cell_alphamesh is a dictionary with structure:
           cell_alphamesh = {'alpha': float, 'mesh': trimesh.Trimesh}

        Args:
            cellID (str)
            alphameshes (dict): dictionary with the alpha-shapes of all cells
            cell_alphamesh (dict): dictionary with the alpha-shape of the current cell
            _*: not used, just to match the signature of the function

        Returns:
            dict: cell-wide alphameshes dictionary updated with the current cell
        """
        alphameshes[cellID] = cell_alphamesh
        return alphameshes
    
    def _nfunc(cellID: str, cte_name: str, _, config: dict) -> dict:
        """ Node function to fit the alpha-shape to a single cell.
        
        Just a wrapper around run_alphashape_single_cell.

        Args:
            cellID (str)
            cte_name (str)
            config (dict): configuration dictionary with the parameters as defined in required_keys
            _: not used, just to match the signature of the function

        Returns:
            cell_alphamesh (dict): dictionary with the alpha-shape of the cell
        """
        
        # Open the CTE
        cte = ChromatinTracingExperiment(cte_name, 'r')
        
        # Unpack the parameters from config
        alpha = config['alpha']
        force = config['force']
        
        # Run the alphashape fitting for the cell
        cell_alphamesh = run_alphashape_single_cell(cellID, cte, alpha, force)
        
        return cell_alphamesh
    
    # Run the alphashape fitting in parallel
    alphameshes = parallel.control_func(
        cte,
        None,
        config,
        required_keys,
        _nfunc,
        _rfunc_init,
        _rfunc_update
    )
    
    return alphameshes



# CONVERSION OF MESH TO IMAGE

def mesh_to_image_single_cell(
    mesh: trimesh.Trimesh, resolution: float, border: int, ndilation: int = None
) -> tuple:
    """ Convert a trimesh.Trimesh mesh to a 3D binary image at a given resolution.

    Args:
        mesh (trimesh.Trimesh): mesh used to create the MRC file
        resolution (float): voxel size of the MRC file (in physical units)
        border (int): black border around the mesh (in voxels)
        ndilation (int, optional): number of dilations to apply to the mask. Defaults to None.

    Returns:
        image (np.array): 3D binary image of the mesh
        origin_vx (tuple): origin of the 3D image in voxel units
        shape (tuple): shape of the 3D image
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
    
    # Use mesh.contains() to create a binary 3D image of the volume
    image = mesh.contains(xyz).reshape(shape).astype(int)
    
    # Dilate the mask to add more depth to the image
    if ndilation is not None:
        image = binary_dilation(image, iterations=ndilation)
    
    # Get the origin of the image in voxel units, so that it matches with the imaging spots
    # It is the first point of the bounding box, quantized by the resolution
    origin_vx = np.round(bbox[0] / resolution).astype(int)
    
    return image, origin_vx, shape

def run_mesh_to_image(cte: ChromatinTracingExperiment, config: dict) -> None:
    """ Parallel code to convert the mesh of each cell to a 3D binary image.
    
    Results are saved in an HDF5 file.
    
    The HDF5 file will have the following structure:
    - The resolution is saved as an attribute of the file.
    - One group for each cell, with the cellID as the group name.
    - Each group will have the following datasets:
        - origin: origin of the MRC in voxel units
        - image: 3D binary image of the mesh
    
    The config specifies the parameters of the calculation:
    - h5_name: name of the output HDF5 file
    - resolution: resolution of the voxel edge (in the same units as the coordinates in CTE)
    - border: white border around the cell nucleus
    - ndilation: number of dilations to apply to the mask
    
    Args:
        cte (ChromatinTracingExperiment)
        config (dict)
    """

    # Define the required keys for the config
    required_keys = {
        'h5_name': {'type': str},
        'resolution': {'type': float, 'positive': True},
        'border': {'type': int, 'positive': True},
        'ndilation': {'type': int, 'positive': True},
    }
    
    def _rfunc_init(_1, _2, _3, _4) -> dict:
        """ Initialization function for reduction step.

        Args:
            _*: not used, just to match the signature of the function
        
        Returns:
            dict: empty dictionary
        """
        return {}
    
    def _rfunc_update(cellID: str, images: dict, cell_image: dict, _1, _2, _3) -> dict:
        """ Update function for reduction step.
        
        images is a dictionary with structure:
              images[cellID] = cell_image
        cell_image is a dictionary with structure:
                cell_image = {'image': np.array, 'origin': np.array}

        Args:
            cellID (str)
            images (dict): dictionary with the images of all cells
            cell_image (dict): dictionary with the image of the current cell
            _*: not used, just to match the signature of the function

        Returns:
            dict: cell-wide images dictionary updated with the current cell
        """
        images[cellID] = cell_image
        return images
    
    def _nfunc(cellID: str, cte_name: str, config: dict) -> dict:
        """ Node function to convert the mesh of a single cell to a 3D binary image.
        
        Just a wrapper around mesh_to_image_single_cell.

        Args:
            cellID (str)
            cte_name (str)
            config (dict)

        Returns:
            dict: dictionary with the image of the cell
        """
        
        # Open the CTE
        cte = ChromatinTracingExperiment(cte_name, 'r')
        # Get the mesh of the cell
        mesh = cte.get_alphashapes(cellID)['mesh']
        
        # Unpack the parameters from config
        resolution = config['resolution']
        border = config['border']
        ndilation = config['ndilation']
        
        # Convert the mesh to an image
        image, origin, shape = mesh_to_image_single_cell(mesh, resolution, border, ndilation)
        
        # Return the image and the origin
        return {'image': image, 'origin': origin}
    
    # Run the mesh to image conversion in parallel
    images = parallel.control_func(
        cte,
        None,
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
        # Save the resolution as an attribute
        f.attrs['resolution'] = config['resolution']
        # Loop over cells
        for cellID, cell_image in images.items():
            # Create a group for the cell
            cell_group = f.create_group(cellID)
            # Save origin, shape in the group (they are the same for all images of the cell)
            cell_group.create_dataset('origin', data=cell_image['origin'])
            cell_group.create_dataset('image', data=cell_image['image'])




# BODIES' VOLUMES KERNEL DENSITY ESTIMATION

def run_bodies_KDE_single_cell(
    cellID: str, cte: ChromatinTracingExperiment, scf: SingleCellFeature,
    voxel_res: float, kde_alpha: float, bodies_to_features: dict,
    border: int = 10,
) -> tuple:
    """ 
    Run the nuclear bodies detection algorithm for a single cell by Kernel Density Estimation.
    
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
    d = cte.get_data(cellID, format='numpy')
    xs, ys, zs = d['xs'], d['ys'], d['zs']
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

def run_bodies_KDE(cte: ChromatinTracingExperiment, scf: SingleCellFeature, config: dict) -> None:
    """ Parallel code to identify nuclear bodies in each cell by Kernel Density Estimation.
    
    Results are saved in an HDF5 file.
    
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
            dict: cell-wide bodies dictionary updated with the current cell
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
        images, origin, bbox, shape =  run_bodies_KDE_single_cell(
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
        # Save the resolution as an attribute
        f.attrs['resolution'] = config['voxel_resolution']
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



# BODIES' VOLUMES BINARIZATION

def run_bodies_binarization_single_cell(
    image: np.ndarray, percentile: float, nerosion: int = 1, ndilation: int = 1
) -> tuple:
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

def run_bodies_binarization(
    bodies_KDE: h5py.File, percentile: float, nerosion: int = 1, ndilation: int = 1
) -> None:
    """ Binarize the continuous-valued KDE images of the nuclear bodies in all cells.
    
    Simply loops over the cells/bodies determined by the bodies_KDE HDF5 file,
    and applies the binarization to each body.
    
    The binarized images are saved in a new HDF5 file.
    
    The HDF5 file will have the following structure:
    - One group for each cell, with the cellID as the group name.
    - Each group will have the following datasets:
        - origin: origin of the MRC in voxel units
        - bbox: bounding box of the MRC
        - shape: shape of the MRC
        - One group for each body, with the body name as the group name.
            - bimage: binarized image of the body
            - limage: labeled image of the body
            - nbodies: number of bodies in the image

    Args:
        bodies_KDE (h5py.File): HDF5 file with the continuous-valued KDE images of the bodies
            (from the run_bodies_KDE function)
        percentile (float): percentile used to binarize the images
            (applied to the KDE non-zero values)
        nerosion (int, optional): number of iterations for the erosion. Defaults to 1.
        ndilation (int, optional): number of iterations for the dilation. Defaults to 1.
    """
    
    # Create a new HDF5 file to save the results
    h5_name = os.path.splitext(bodies_KDE.filename)[0] + '_binarized.h5'
    h5 = h5py.File(h5_name, 'w')
    
    # Save the resolution (from the original HDF5 file) as an attribute
    h5.attrs['resolution'] = bodies_KDE.attrs['resolution']
    
    # Loop over the cells
    for cellID in bodies_KDE:
        
        # Create a group for the cell
        cell_group = h5.create_group(cellID)
        
        # Store the origin, bbox, and shape
        cell_group.create_dataset('origin', data=bodies_KDE[cellID]['origin'])
        cell_group.create_dataset('bbox', data=bodies_KDE[cellID]['bbox'])
        cell_group.create_dataset('shape', data=bodies_KDE[cellID]['shape'])
        
        # Loop over the bodies
        for body in bodies_KDE[cellID]:
            if body in ['origin', 'bbox', 'shape']:
                continue
            
            # Get the image of the body
            image = bodies_KDE[cellID][body][:]
            
            # Binarize the image
            bimage, limage, nbodies = run_bodies_binarization_single_cell(
                image, percentile, nerosion, ndilation
            )
            
            # Create a group for the body
            body_group = cell_group.create_group(body)
            
            # Save the binarized, labeled image and the number of bodies
            body_group.create_dataset('bimage', data=bimage)
            body_group.create_dataset('limage', data=limage)
            body_group.create_dataset('nbodies', data=nbodies)
    
    h5.close()


# UTILITY FUNCTIONS

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
