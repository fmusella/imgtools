# Functions for creating files for visualization: cmm, pdb, mrc.

import os
import numpy as np
import pickle
from alabtools.plots import write_pdb
from .cte import ChromatinTracingExperiment
from . import cte_utils
from . import parallelization
from .metrics import get_trace_ranks_for_cell
from .. import utils


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
    
    for cellID in cte.cell_labels:
        save_cell_pdb(cte, cellID, path)


# CMM

def save_cell_cmm(cte: ChromatinTracingExperiment, cellID: str, path: str, radius: float) -> None:
    """ Write a cmm file for a cell.
    
    Each trace is written in a separate cmm file.

    Args:
        ct (ChromatinTracingExperiment)
        cellID (str)
        path (str): directory where the cmm files will be saved.
        radius (float): size of the markers (in physical units)
    """
    
    # Check that the path exists. If not, create it.
    if not isinstance(path, str):
        raise TypeError("path must be a string.")
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Get the data for the cell in dictionary format
    cell_data = cte.get_data(cellID)
    
    # Loop over chromosomes and traces, and write each trace to a separate cmm file
    for chrom in cell_data:
        for traceID in cell_data[chrom]:
            
            xs, ys, zs, starts, ends, lums, spotIDs = cte_utils.trace_dict_to_numpy(cell_data[chrom][traceID])
            
            utils.write_cmm(
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
    origin, shape = _mrc_nfunc(cellID, None, None, None, cte.get_alphashapes(cellID), config)
    
    return origin, shape

mrc_required_keys = {
    'resolution': {'type': float, 'positive': True},
    'border': {'type': int, 'positive': True},
    'surface_thickness': {'type': float, 'positive': True},
    'mrc_path': {'type': str}
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
    origin, shape = utils.mesh_to_mrc(
        path = config['mrc_path'],
        name_prefix = cellID,
        mesh = alphashape['mesh'],
        resolution = config['resolution'],
        border = config['border'],
        surface_thickness = config['surface_thickness']
    )
    
    cell_mrc_params = {'origin': origin, 'shape': shape}
    
    return cell_mrc_params
