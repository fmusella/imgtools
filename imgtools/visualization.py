import os
import numpy as np
import trimesh
import mrcfile
from matplotlib import pyplot as plt


# Pyplot functions

def cell_pyplot_default_params(cellID):
    """ Default parameters for pyplot cell plots.

    Returns:
        params (dict): Default parameters for pyplot cell plots.
    """
    
    params = {
        'figsize': (10, 10),
        'dpi': 200,
        'show_title': False,
        'show_axis': True,
        'show_legend': False,
        'title': 'Cell ' + str(cellID)
    }
    
    return params

def cell_pyplot_complete_params(cellID: str, params: dict):
    """ Complete parameters for pyplot cell plots.

    Args:
        cellID (str)
        params (dict): Incomplete parameters for pyplot cell plots.

    Returns:
        (dict): Complete parameters for pyplot cell plots.
    """
    
    # Get default parameters
    default_params = cell_pyplot_default_params(cellID)
    
    # If params does not contain a key, add it from default_params
    for key in default_params.keys():
        if key not in params.keys():
            params[key] = default_params[key]
    
    return params

def cell_pyplot(filename: str, cellID: str, data: dict, params: dict):
    """ Plot cell data using pyplot.

    Args:
        filename (str): destination filename
        cellID (str): cell ID
        data (dict): data to plot
        params (dict): parameters for pyplot cell plots
    """
    
    # Check input data
    if not isinstance(data, dict):
        raise ValueError('data must be a dict')
    if 'x' not in data.keys():
        raise ValueError('data must contain a key "x"')
    if 'y' not in data.keys():
        raise ValueError('data must contain a key "y"')
    if 'z' not in data.keys():
        raise ValueError('data must contain a key "z"')
    if 'chrom' not in data.keys():
        raise ValueError('data must contain a key "chrom"')
    if len(data['x']) != len(data['y']) or len(data['x']) != len(data['z']) or len(data['x']) != len(data['chrom']):
        raise ValueError('data["x"], data["y"], data["z"] and data["chrom"] must have the same length')
    
    # Complete parameters
    params = cell_pyplot_complete_params(cellID, params)
    
    # Create 3D figure
    fig = plt.figure(figsize=params['figsize'], dpi=params['dpi'])
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot data, each chromosome in a different color
    for i, chrom in enumerate(np.unique(data['chrom'])):
        
        # Get chromosome data
        x_chrom = data['x'][data['chrom'] == chrom]
        y_chrom = data['y'][data['chrom'] == chrom]
        z_chrom = data['z'][data['chrom'] == chrom]

        # Plot chromosome data
        ax.scatter(x_chrom, y_chrom, z_chrom, label=chrom, color='C' + str(i))
    
    # Set legend
    if params['show_legend']:
        ax.legend(loc='best')
    
    # Set title
    if params['show_title']:
        ax.set_title(params['title'])
    
    # Set axis
    if params['show_axis']:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    else:
        ax.set_axis_off()
    
    # Save figure in 3 different angles: parallel to xy, parallel to xz and parallel to yz
    ax.view_init(0, 0)
    plt.savefig(filename + '_xy.png')
    ax.view_init(90, 0)
    plt.savefig(filename + '_xz.png')
    ax.view_init(0, 90)
    plt.savefig(filename + '_yz.png')
    
    plt.close(fig)


# SCRIPTS TO SAVE MRC FILES

def mesh_to_mrc(
    path: str,
    name_prefix: str,
    mesh: trimesh.Trimesh,
    resolution: float,
    border: int, 
    surface_thickness: float  
):
    """ Save a mesh as a MRC file.

    Args:
        path (str): directory where the MRC file will be saved
        name_prefix (str): prefix of the MRC file name
        mesh (trimesh.Trimesh): mesh used to create the MRC file
        resolution (float): voxel size of the MRC file (in physical units)
        border (int): black border around the mesh (in voxels)
        surface_thickness (float): thickness of the surface (in physical units)

    Returns:
        origin_mrc_vx (tuple): origin of the MRC file in voxel units
        shape (tuple): shape of the MRC file
    """
    
    # Check if the path exists
    if not isinstance(path, str):
        raise TypeError('path must be a string')
    if not os.path.exists(path):
        raise NotADirectoryError('path does not exist')
    
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
    
    # Get the origin of the mrc file in voxel units
    # The negative sign is because ?????
    origin_mrc_vx = - np.round(bbox[0] / resolution).astype(int)
    
    # Save the volume mask as a MRC file
    write_mrc(
        filename = os.path.join(path, name_prefix + '.mrc'),
        data = volume_mask,
        origin = tuple(origin_mrc_vx),
        voxel_size = (resolution, resolution, resolution)
    )
    
    # Compute the surface of the mask
    surface_dists = trimesh.proximity.signed_distance(mesh, xyz).reshape(shape)
    surface_mask = (np.abs(surface_dists) <= surface_thickness).astype(int)
    
    # Save the surface mask as a MRC file
    write_mrc(
        filename = os.path.join(path, name_prefix + '_surface.mrc'),
        data = surface_mask,
        origin = tuple(origin_mrc_vx),
        voxel_size = (resolution, resolution, resolution)
    )
    
    del volume_mask, surface_mask, surface_dists, xyz, x, y, z, bbox
    
    return origin_mrc_vx, shape

def write_mrc(filename, data, origin=(0, 0, 0), voxel_size=(1, 1, 1)):
    """Write a MRC file from a numpy array.

    Args:
        filename (str): name of the file to be written.
        data (np.array(shape=(n_x_grid, n_y_grid, n_z_grid))): grid of values (0 or 1)
    """
    # Swap the axes to match the MRC format
    data = np.swapaxes(data, 0, 2)
    # Ensure the data is in int8 format as we'll use MODE 0
    data = data.astype(np.int8)
    # Create a new MRC file and save the data
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.nstart = origin
        mrc.voxel_size = voxel_size

def create_grid(bbox: np.array, resolution: float):
    """ Create a 3D grid of points.

    Args:
        bbox (np.array):
            array of shape (2, 3) containing the min and max values of the bounding box
        resolution (float): resolution of the grid
    """
    x = np.arange(bbox[0, 0], bbox[1, 0], resolution)
    y = np.arange(bbox[0, 1], bbox[1, 1], resolution)
    z = np.arange(bbox[0, 2], bbox[1, 2], resolution)
    # Crate an array of shape (M, 3) where M = len(x) * len(y) * len(z),
    # and each row is a point in the 3D grid [x, y, z],
    # where x, y, z span all the possible combinations of values in x, y, z
    xyz = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)  # TODO: check if this is correct
    shape = (len(x), len(y), len(z))
    assert xyz.shape[0] == len(x) * len(y) * len(z)
    assert xyz[0] == [x[0], y[0], z[0]]
    return xyz, shape
