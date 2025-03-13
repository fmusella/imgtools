import numpy as np
from scipy.stats import gaussian_kde
from scipy.ndimage import binary_dilation, binary_erosion, label
import trimesh
from .cte import ChromatinTracingExperiment
from .scf import SingleCellFeature

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

def binarize(
    image: np.ndarray, percentile: float, nerosion: int, ndilation: int,
    voxel_res: float, dust_threshold: float = None
):
    # Now binarize the image
    # Calculate the threshold of binarization as a percentile of the KDE intensities
    # (we exclude the 0 values)
    threshold = np.nanpercentile(image[image > 0], percentile)
    bimage = (image >= threshold).astype(int)
    # Perform erosion and dilation to remove small objects and fill holes
    bimage = binary_erosion(bimage, iterations=nerosion)
    bimage = binary_dilation(bimage, iterations=ndilation)
    # Now label the connected components
    bimage, nbodies = label(bimage)
    # Remove the small objects if dust_threshold is not None
    if dust_threshold is not None:
        for b in range(1, nbodies + 1):
            b_volume = np.sum(bimage == b) * voxel_res**3
            if b_volume < dust_threshold:
                bimage[bimage == b] = 0
    # Relabel the connected components
    bimage, _ = label(bimage)
    return bimage
