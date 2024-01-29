# Functions for creating files for visualization: cmm, pdb, mrc.

import os
import numpy as np
import trimesh
from alabtools.plots import write_pdb
from .cte import ChromatinTracingExperiment
from . import cte_utils
from .metrics import get_trace_ranks_for_cell


# PDB

def save_cell_pdb(cte: ChromatinTracingExperiment, cellID: str, path: str, filename: str = None) -> None:
    """Write a pdb file for a cell.
    The noise traces are not written."""
    
    # Check that cellID is a string and that it is in the data
    if not isinstance(cellID, str):
        raise TypeError("cellID must be a string.")
    if not cellID in cte.data:
        raise ValueError("cellID {} not in data.".format(cellID))
    
    if not isinstance(path, str):
        raise TypeError("path must be a string.")
    if not os.path.exists(path):
        raise NotADirectoryError("Directory {} does not exist.".format(path))
    
    # Get data for cell in numpy array format
    xs, ys, zs, chroms, starts, _, lums, traceIDs, _ = cte_utils.cell_to_numpy(cte.data[cellID])
    
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
    
    # Convert start to units of 100000 bp, so that it fits in the occupancy field of the pdb file
    # i.e. 200000000 bp --> 2000.00
    starts = starts / 100000
    
    # Convert lums so that they fit in the beta field of the pdb file
    lums = lums - np.min(lums)
    lums = lums / np.max(lums)
    lums = lums * 1000
    
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
    
    assert isinstance(path, str), "path must be a string."
    assert os.path.exists(path), "path does not exist."
    
    for cellID in cte.data:
        save_cell_pdb(cellID, path)


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

required_keys_mrc = {
    'resolution': {'type': float, 'positive': True},
    'border': {'type': int, 'positive': True},
    'surface_thickness': {'type': float, 'positive': True},
    'mrc_path': {'type': str},
}

def do_cell_mrc(cellID: str, cell_mesh: trimesh.Trimesh, params: dict):
    """ Generate the mrc file for a single cell.

    Args:
        cellID (str): The cell ID.
        cell_mesh (trimesh.Trimesh): The alphashape of the cell.
        params (dict): Parameters for the mrc file generation task.

    Returns:
        origin (tuple): Origin of the mrc file in voxel units.
        shape (tuple): Shape of the mrc file in voxel
    """
    
    origin, shape = plots.mesh_to_mrc(
        path=params['mrc_path'],
        name_prefix=cellID,
        mesh=cell_mesh,
        resolution=params['resolution'],
        border=params['border'],
        surface_thickness=params['surface_thickness']
    )
    
    return origin, shape

def mrc_parallel(cellID: str, config: dict, tempdir: str):
    """Parallel function for the alphashape task.

    Args:
        cellID (str): The cell ID.
        config (dict): The config file for the alphashape task.
        tempdir (str): Temporary directory for storing intermediate results.
    """
    
    check_config(config, required_keys_mrc)
    
    assert isinstance(cellID, str), "cellID {} should be a string. Got type: {}".format(cellID, type(cellID))
    
    assert isinstance(tempdir, str), "tempdir should be a string. Got type: {}".format(type(tempdir))
    assert os.path.isdir(tempdir), "tempdir is not a valid directory."
    
    # Load file with the alphashape for the cell
    in_filename = os.path.join(tempdir, '{}_mesh.pickle'.format(cellID))
    
    assert os.path.isfile(in_filename), "Mesh for cell {} not found.".format(cellID)
    
    with open(in_filename, 'rb') as f:
        cell_mesh = pickle.load(f)
    
    # Write the mrc file for the cell
    origin, shape = do_cell_mrc(cellID, cell_mesh, config)
    
    del cell_mesh
    
    # Save the origin and shape for the cell with pickle
    out_filename = os.path.join(tempdir, '{}_mrc_params.pickle'.format(cellID))
    with open(out_filename, 'wb') as f:
        pickle.dump({'origin': origin, 'shape': shape}, f)
    
    return cellID

def mrc_reduce(cellIDs: list, config: dict, tempdir: str):
    """ Reduce function for the mrc file generation task.
    
    Collects the parameters of the mrc files for all cells.

    Args:
        cellIDs (list): list of cell IDs.
        tempdir (str): temporary directory for storing intermediate results.

    Returns:
        mrc_params (dict): Dictionary of mrc parameters for all cells in dictionary format.
    """
    
    # Check cellIDs
    assert isinstance(cellIDs, list), "cellIDs should be a list. Got type: {}".format(type(cellIDs))
    assert len(cellIDs) > 0, "cellIDs should not be empty."
    
    # Initialize the output, which is a dictionary of parameters for the mrc files
    mrc_params = {}

    for cellID in cellIDs:
        
        # Get the filename for the mrc parameters of the cell
        filename = os.path.join(tempdir, '{}_mrc_params.pickle'.format(cellID))
        
        # Check that the file exists
        assert os.path.isfile(filename), "MRC param file for cell {} not found.".format(cellID)

        # Load the file
        with open(filename, 'rb') as f:
            cell_mrc_params = pickle.load(f)
        
        # Get the data from the pickle file
        origin = cell_mrc_params['origin']
        shape = cell_mrc_params['shape']
        
        # Add the data to the output
        mrc_params[cellID] = {
            'origin': origin,
            'shape': shape
        }
    
    # Save the mrc parameters for all cells with pickle
    out_filename = os.path.join(config['mrc_path'], 'mrc_params.pickle')
    with open(out_filename, 'wb') as f:
        pickle.dump(mrc_params, f)

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
    
    del volume_mask, surface_mask, surface_dists, xyz, bbox
    
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

def run_mrc(cte: ChromatinTracingExperiment, config: dict):
    # MOVE TO VOLUMES.PY
    """ Performs the mrc file creation task on the population.
    
    The mrc files (volumes and surfaces) are stored in the path specified in config.
    
    The function also saves - in this path - a pickle file with the origins and shapes
    of each cell volume.
    
    Args:
        config (dict): configuration dictionary for the mrc file creation.
    """
    
    # Create a temporary directory
    tempdir = tempfile.mkdtemp(dir=os.getcwd())
    sys.stdout.write("Temporary directory for nodes' results: {}\n".format(tempdir))
    
    # Save the data of each cell separately in the temporary directory as a pickle file
    for cellID in cte.alphashapes:
        filename = os.path.join(tempdir, '{}_mesh.pickle'.format(cellID))
        with open(filename, 'wb') as f:
            pickle.dump(cte.alphashapes[cellID]['mesh'], f)
    
    # set the parallel and reduce tasks
    parallel_task = partial(parallelization.mrc_parallel, config=config, tempdir=tempdir)
    reduce_task = partial(parallelization.mrc_reduce, config=config, tempdir=tempdir)
    
    # create a Controller
    controller = Controller(config)

    # run the parallel task
    controller.map_reduce(parallel_task, reduce_task, args=list(cte.alphashapes.keys()))
    
    # Delete the non-empty temporary directory
    os.system('rm -r {}'.format(tempdir))
    
    del controller

def run_mrc_single_cell(cte: ChromatinTracingExperiment, cellID: str, params: dict):
    # MOVE TO VOLUMES.PY
    """ Performs the mrc file creation task on a single cell.
    
    The mrc files (volume and surface) are stored in the path
    specified in params.
    
    The function returns the origin and shape of the volume mrc file,
    necessary for aligning the mrc files in 3D space.

    Args:
        cellID (str): cell ID.
        params (dict): configuration dictionary for the mrc file creation.

    Returns:
        origin (tuple): origin of the volume mrc file in voxel units.
        shape (tuple): shape of the volume mrc file in voxel units.
    """
    
    # Check that all required keys are present in params
    parallelization.check_config(params, parallelization.required_keys_mrc, parallel=False)
    
    # Perform the mrc file creation
    origin, shape = parallelization.do_cell_mrc(cellID, cte.alphashapes[cellID]['mesh'], params)
    
    return origin, shape
