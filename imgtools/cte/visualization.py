# Functions for creating files for visualization: cmm, pdb, mrc.

import os
import numpy as np
import trimesh
import mrcfile
import pickle
from alabtools.plots import write_pdb
from .cte import ChromatinTracingExperiment
from . import cte_utils
from . import parallelization
from .metrics import get_trace_ranks_for_cell


# PDB

def save_cell_pdb(cte: ChromatinTracingExperiment, cellID: str, path: str, filename: str = None) -> None:
    """Write a pdb file for a cell.
    The noise traces are not written."""
    
    # Check that the path exists. If not, create it.
    if not isinstance(path, str):
        raise TypeError("path must be a string.")
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Get data for cell in numpy array format
    xs, ys, zs, chroms, starts, ends, lums, traceIDs, spotIDs = cte.get_data(cellID, format='numpy')
    
    # Convert chroms to chromnums, e.g. 'chr1' --> '1', 'chrX' --> 'X'
    chromnums = []
    for c in chroms:
        chromnums.append(c.replace('chr', ''))
    chromnums = np.array(chromnums).astype('U20')

    # Convert traceIDs to trace ranks within each chromosome, and then to strings
    # e.g. traceID: '12_1' --> trace_rank: 1 ---> tracenum: 'A'
    tranks = get_trace_ranks_for_cell(cte, cellID)  # ranks of each trace in each chromosome of the cell
    tracenums = []
    for chrom, traceID in zip(chroms, traceIDs):
        t = tranks[chrom][traceID]  # rank of traceID in chrom
        if t > 0:
            # Valid traces (positive integers) are converted like this:
            #   1 --> 'A', 2 --> 'B', ...
            tracenums.append(chr(t + 64))
        elif t < 0:
            # Noisy traces (negative integers) are converted like this:
            #   -1 --> 'Z', -2 --> 'Y', ...
            tracenums.append(chr(t + 91))
        else:
            raise Exception("Trace number cannot be 0.")
    tracenums = np.array(tracenums).astype('U20')
    
    # Convert starts to units in bp such that the maximum values has 4 digits above the decimal point (i.e. < 10000)
    while np.max(starts) >= 10000:
        starts = starts / 10
    # Truncate to 2 decimal places
    starts = np.round(starts, 2)
    
    # Convert lums so that the minimum value is 0 and the maximum value is 1000
    lums = lums - np.min(lums)
    lums = lums / np.max(lums)
    lums = lums * 1000
    # Truncate to 2 decimal places
    lums = np.round(lums, 2)
    
    # Write dictionary for pdb file
    celldata_for_pdb = {
        'x': xs,
        'y': ys,
        'z': zs,
        'residue_name': chromnums,
        'chain_id': tracenums,
        'occupancy': starts,
        'beta': lums
    }
    
    # Write pdb file
    if filename is None:
        filename = os.path.join(path, cellID + '.pdb')
    
    write_pdb(filename, celldata_for_pdb)

def save_all_pdbs(cte: ChromatinTracingExperiment, path: str) -> None:
    """Write pdb files for all cells."""
    
    for cellID in cte.get_cellIDs():
        save_cell_pdb(cte, cellID, path)


# CMM

def write_cmm(filename: str, marker_str: str, coord: np.ndarray, radius: float, color: np.ndarray = [0, 0, 0]) -> None:
    """ Write a CMM file.
    
    Only works for a single marker set. Colors all markers and links with the same color.

    Args:
        filename (str): name of the file to be written
        marker_str (str): string to identify the marker set
        coord (np.ndarray): numpy array of shape (n_markers, 3) containing the coordinates of the markers
        radius (float): size of the markers (in physical units)
        color (np.ndarray, optional): numpy array of shape (3,) containing the RGB color of the markers and links. Defaults to [0, 0, 0].
    """

    with open(filename,'w') as f:
        
        f.write('<marker_set name="marker set %s">\n' % marker_str)
        
        # Write markers
        for i in range(len(coord)):
            f.write(
                '<marker id="%d" x="%.3f" y="%.3f" z="%.3f" r="%.3f" g="%.3f" b="%.3f" radius="%.3f" note="" nr="%.3f" ng="%.3f" nb="%.3f"/>\n'
                    % (i + 1, coord[i, 0], coord[i, 1], coord[i, 2], color[0], color[1], color[2], radius, color[0], color[1], color[2])
            )
        
        # Write links
        for i in range(len(coord) - 1):
            f.write(
                '<link id1="%d" id2="%d" r="%.3f" g="%.3f" b="%.3f" radius="%.3f" />\n'
                    % (i + 1, i + 2, color[0], color[1], color[2], radius)
            )
        
        f.write('</marker_set>\n')

def save_cell_cmm(cte: ChromatinTracingExperiment, cellID: str, path: str, radius: float) -> None:
    """ Write a cmm file for a cell.
    
    Each trace is written in a separate cmm file.

    Args:
        ct (ChromatinTracingExperiment)
        cellID (str)
        path (str): directory where the cmm files will be saved.
        radius (float): size of the markers (in physical units)
    """
    
    if cellID not in cte.data:
        raise ValueError("cellID {} not in data.".format(cellID))
    
    if not os.path.exists(path):
        raise NotADirectoryError("Directory {} does not exist.".format(path))
    
    for chrom in cte.data[cellID]:
        for traceID in cte.data[cellID][chrom]:
            
            xs, ys, zs, _, _, _, _, _ = cte_utils.trace_dict_to_numpy(cte.data[cellID][chrom][traceID])
            
            write_cmm(
                filename = os.path.join(path, '{}_{}_{}.cmm'.format(cellID, chrom, traceID)),
                marker_str = 'cellID: {}, chrom: {}, traceID: {}'.format(cellID, chrom, traceID),
                coord = np.array([xs, ys, zs]).T,
                radius = radius,
            )


# MRC

def run_mrc(cte: ChromatinTracingExperiment, config: dict) -> None:
    """ Creates the mrc files for all cells in the experiment in parallel.
    The files are created in a folder specified in config, together with a pickle file
    containing the origin and shape of each MRC file.

    Args:
        cte (ChromatinTracingExperiment)
        config (dict): configuration dictionary for the mrc file creation
    """
    
    def _rfunc_init(_1, _2, _3, _4, _5) -> dict:
        """ Initialize the mrc parameters dictionary for the reduce function.

        Args:
            _*: not used, just to match the signature of the function

        Returns:
            mrc_params (dict): empty dictionary
        """
        mrc_params = {}
        return mrc_params
    
    def _rfunc_update(cellID: str, mrc_params: dict, cell_mrc_params: dict, _2, _3, _4, _5, _6) -> dict:
        """ Update the mrc parameters dictionary for the reduce function.

        Args:
            cellID (str)
            mrc_params (dict): mrc parameters dictionary for the entire population
            cell_mrc_params (dict): mrc parameters dictionary for the cell
            _*: not used, just to match the signature of the function

        Returns:
            mrc_params (dict): updated mrc parameters dictionary for the entire population
        """
        mrc_params[cellID] = cell_mrc_params
        return mrc_params
    
    # Run the MRC calculation in parallel
    # The MRC files are saved in the folder specified in config, and here we return the origin and shape of each cell
    mrc_params = parallelization.control_func(
        cte,
        config,
        mrc_required_keys,
        _mrc_nfunc,
        _rfunc_init,
        _rfunc_update
    )
    
    # Save the mrc parameters as a pickle file in the folder specified in config
    out_filename = os.path.join(config['mrc_path'], 'mrc_params.pickle')
    with open(out_filename, 'wb') as f:
        pickle.dump(mrc_params, f)
    
    del mrc_params

def run_mrc_single_cell(cte: ChromatinTracingExperiment, cellID: str, config: dict) -> tuple:
    """ Performs the mrc file creation task on a single cell.
    
    The mrc files (volume and surface) are stored in the path
    specified in config.
    
    The function returns the origin and shape of the volume mrc file,
    necessary for aligning the mrc files in 3D space.

    Args:
        cellID (str): cell ID.
        config (dict): configuration dictionary for the mrc file creation.

    Returns:
        origin (tuple): origin of the volume mrc file in voxel units.
        shape (tuple): shape of the volume mrc file in voxel units.
    """
    
    # Check that all required keys are present in config
    parallelization.check_config(config, mrc_required_keys, parallel=False)
    
    # Perform the mrc file creation
    origin, shape = _mrc_nfunc(cellID, None, None, None, cte.alphashapes[cellID]['mesh'], config)
    
    return origin, shape

mrc_required_keys = {
    'resolution': {'type': float, 'positive': True},
    'border': {'type': int, 'positive': True},
    'surface_thickness': {'type': float, 'positive': True},
    'mrc_path': {'type': str},
    'use': {'data': False, 'index': False, 'alphashapes': True}
}

def _mrc_nfunc(cellID: str, _1, _2, _3, alphashape: dict, config: dict) -> dict:
    """ Node function to save the cell MRC file.
    Saves the MRC file for the cell and returns the origin and shape of the file.

    Args:
        cellID (str)
        alphashape (dict): alphashape dictionary for the cell
                           alphashape['alpha']: float, alpha parameter used to compute the alphashape
                           alphashape['mesh']: trimesh.Trimesh, mesh of the alphashape
        config (dict): configuration dictionary for the mrc file creation

    Returns:
        cell_mrc_params (dict): dictionary with the origin and shape of the cell MRC file
                                cell_mrc_params['origin']: tuple, origin of the cell MRC file in voxel units
                                cell_mrc_params['shape']: tuple, shape of the cell MRC file in voxel units
    """
    
    # Save the mrc file for the cell and return the origin and shape of the file
    origin, shape = mesh_to_mrc(
        path = config['mrc_path'],
        name_prefix = cellID,
        mesh = alphashape['mesh'],
        resolution = config['resolution'],
        border = config['border'],
        surface_thickness = config['surface_thickness']
    )
    
    cell_mrc_params = {'origin': origin, 'shape': shape}
    
    return cell_mrc_params

def mesh_to_mrc(
    path: str,
    name_prefix: str,
    mesh: trimesh.Trimesh,
    resolution: float,
    border: int, 
    surface_thickness: float  
) -> tuple:
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
    
    del volume_mask, surface_mask, surface_dists, xyz, bbox
    
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
    # Swap the axes to match the MRC format
    data = np.swapaxes(data, 0, 2)
    # Ensure the data is in int8 format as we'll use MODE 0
    data = data.astype(np.int8)
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
