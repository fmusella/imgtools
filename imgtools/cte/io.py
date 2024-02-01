# Functions for saving and reading data in the ChromatinTracingExperiment objects.

import numpy as np
import h5py
from . import cte_utils


# SAVE FUNCTIONS

def save_data_to_hdf5(data: dict, f: h5py.File) -> None:
    """ Save the CTE data to an hdf5 file.
    
    The data is saved in numpy format, in the group 'data', with a subgroup for each cellID.
    
    Each cell subgroup contains the following datasets:
        'xs', 'ys', 'zs', 'chroms', 'starts', 'ends', 'lums', 'traceIDs', 'spotIDs'.

    Args:
        data (dict): dictionary with the data.
        f (h5py.File): hdf5 file.
    """
    
    # Create a dataset to store the cellIDs
    cellIDs = np.array(list(data.keys()), dtype=np.dtype('S20'))
    f.create_dataset('cellIDs', data=cellIDs)
    
    # Create a group for the data
    data_group = f.create_group('data')
    
    # Loop over the cellIDs and save the data in the group
    for cellID in data:
        
        # Convert the cell data from dictionary to numpy format
        xs, ys, zs, chroms, starts, ends, lums, traceIDs, spotIDs = cte_utils.cell_dict_to_numpy(data[cellID])
        
        # Create a group for the cell data
        cell_group = data_group.create_group(cellID)
        
        # Save the cell data in the group
        cell_group.create_dataset('xs', data=xs)
        cell_group.create_dataset('ys', data=ys)
        cell_group.create_dataset('zs', data=zs)
        cell_group.create_dataset('chroms', data=chroms.astype('S10'), dtype=np.dtype('S10'))
        cell_group.create_dataset('starts', data=starts)
        cell_group.create_dataset('ends', data=ends)
        cell_group.create_dataset('lums', data=lums)
        cell_group.create_dataset('traceIDs', data=traceIDs.astype('S20'), dtype=np.dtype('S20'))
        cell_group.create_dataset('spotIDs', data=spotIDs.astype('S20'), dtype=np.dtype('S20'))

def save_attrs_to_hdf5(attrs: dict, f: h5py.File) -> None:
    """ Save the attributes to an hdf5 file.

    Args:
        attrs (dict): dictionary with the attributes.
        f (h5py.File): hdf5 file.
    """
    
    # Save the attributes in the group
    for key in attrs:
        f.attrs[key] = attrs[key]

def save_cell_states_to_hdf5(cell_states: dict, f: h5py.File) -> None:
    pass


# LOAD FUNCTIONS

def load_cell_data_from_hdf5(cellID: str, f: h5py.File, format: str = 'dict'):
    """ Load the CTE data from an hdf5 file.

    Args:
        cellID (str)
        f (h5py.File)
        format (str): Output format of the data. Options: 'dict', 'numpy'.

    Returns:
        data (dict or tuple): dictionary with the data.
    """
        
    # Load the cell data
    cell_group = f['data'][cellID]
    xs = cell_group['xs'][:]
    ys = cell_group['ys'][:]
    zs = cell_group['zs'][:]
    chroms = cell_group['chroms'][:].astype('U10')
    starts = cell_group['starts'][:]
    ends = cell_group['ends'][:]
    lums = cell_group['lums'][:]
    traceIDs = cell_group['traceIDs'][:].astype('U20')
    spotIDs = cell_group['spotIDs'][:].astype('U20')
    
    # Convert the cell data from numpy to dictionary format
    if format == 'dict':
        data = cte_utils.cell_numpy_to_dict(xs, ys, zs, chroms, starts, ends, lums, traceIDs, spotIDs)
    else:
        data = (xs, ys, zs, chroms, starts, ends, lums, traceIDs, spotIDs)
    
    return data

def load_chrom_data_from_hdf5(cellID: str, chrom: str, f: h5py.File, format: str = 'dict'):
    """ Load the CTE data from an hdf5 file.

    Args:
        cellID (str)
        chrom (str)
        f (h5py.File)
        format (str): Output format of the data. Options: 'dict', 'numpy'.

    Returns:
        data (dict or tuple): dictionary with the data.
    """
        
    # Load the cell data
    xs, ys, zs, chroms, starts, ends, lums, traceIDs, spotIDs = load_cell_data_from_hdf5(cellID, f, format='numpy')
    
    # Select the data for the specified chromosome
    idx = np.where(chroms == chrom)[0]
    xs = xs[idx]
    ys = ys[idx]
    zs = zs[idx]
    starts = starts[idx]
    ends = ends[idx]
    lums = lums[idx]
    traceIDs = traceIDs[idx]
    spotIDs = spotIDs[idx]
    
    # Convert the cell data from numpy to dictionary format
    if format == 'dict':
        data = cte_utils.chrom_numpy_to_dict(chrom, xs, ys, zs, starts, ends, lums, traceIDs, spotIDs)
    else:
        data = (xs, ys, zs, starts, ends, lums, traceIDs, spotIDs)
    
    return data

def load_trace_data_from_hdf5(cellID: str, chrom: str, traceID: str, f: h5py.File, format: str = 'dict'):
    """ Load the CTE data from an hdf5 file.

    Args:
        cellID (str)
        chrom (str)
        traceID (str)
        f (h5py.File)
        format (str): Output format of the data. Options: 'dict', 'numpy'.

    Returns:
        data (dict or tuple): dictionary with the data.
    """
        
    # Load the cell data
    xs, ys, zs, chroms, starts, ends, lums, traceIDs, spotIDs = load_chrom_data_from_hdf5(cellID, chrom, f, format='numpy')
    
    # Select the data for the specified traceID
    idx = np.where(np.logical_and(chroms == chrom, traceIDs == traceID))[0]
    xs = xs[idx]
    ys = ys[idx]
    zs = zs[idx]
    starts = starts[idx]
    ends = ends[idx]
    lums = lums[idx]
    spotIDs = spotIDs[idx]
    
    # Convert the cell data from numpy to dictionary format
    if format == 'dict':
        data = cte_utils.cell_numpy_to_dict(xs, ys, zs, chroms, starts, ends, lums, traceIDs, spotIDs)
    else:
        data = (xs, ys, zs, chroms, starts, ends, lums, traceIDs, spotIDs)
    
    return data
