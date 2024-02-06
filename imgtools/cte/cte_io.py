# Functions for saving and reading data in the ChromatinTracingExperiment objects.

import numpy as np
import h5py
import trimesh
from alabtools.utils import Index
from . import cte_utils


# SAVE/LOAD INDEX

def save_index_to_hdf5(index: Index, f: h5py.File) -> None:
    """ Save the index to an hdf5 file. """
    index.save(f)

def load_index_from_hdf5(f: h5py.File) -> Index:
    """ Load the index from an hdf5 file. """
    index = Index(f)
    return index


# SAVE/LOAD ATTRIBUTES

def save_attrs_to_hdf5(attrs: dict, f: h5py.File) -> None:
    """ Save the attributes to an hdf5 file.
    The attributes are saved in the root group. """
    for key in attrs:
        f.attrs[key] = attrs[key]

def load_attrs_from_hdf5(f: h5py.File) -> dict:
    """ Load the attributes from an hdf5 file. """
    attrs = {}
    for key in f.attrs:
        attrs[key] = f.attrs[key]
    return attrs


# SAVE/LOAD CELL LABELS

def save_cell_labels_to_hdf5(cell_labels: list, f: h5py.File) -> None:
    """ Save the cell_labels array to an hdf5 file.
    The cell_labels are saved as an array of 'S20' (string of 20 characters). """
    f.create_dataset('cell_labels', data=np.array(cell_labels).astype('S20'), dtype=np.dtype('S20'))

def load_cell_labels_from_hdf5(f: h5py.File) -> np.ndarray:
    """ Load the cell_labels array from an hdf5 file.
    The cell_labels are loaded as an array of 'U20' (unicode string of 20 characters)."""
    cell_labels = f['cell_labels'][:].astype('U20')
    return cell_labels


# SAVE/LOAD DATA

def save_data_to_hdf5(data: dict, f: h5py.File) -> None:
    """ Save the CTE data to an hdf5 file.
    
    The data is saved in numpy format, in the group 'data', with a subgroup for each cellID.
    
    Each cell subgroup contains the following datasets:
        'xs', 'ys', 'zs', 'chroms', 'starts', 'ends', 'lums', 'traceIDs', 'spotIDs'.

    Args:
        data (dict): dictionary with the data.
        f (h5py.File): hdf5 file.
    """
    # Create a group for the data
    data_group = f.create_group('data')
    # Loop over the cell_labels and save the data in the group
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
        data = cte_utils.trace_numpy_to_dict(xs, ys, zs, starts, ends, lums, spotIDs)
    else:
        data = (xs, ys, zs, chroms, starts, ends, lums, spotIDs)
    
    return data


# SAVE/LOAD ALPHASHAPES

def save_alphashapes_to_hdf5(alphashapes: dict, f: h5py.File) -> None:
    """ Save the alphashapes to an hdf5 file.

    Args:
        alphashapes (dict): dictionary with the alphashapes.
                            alphashapes[cellID] = {'alpha': float, 'mesh': trimesh.Trimesh}.
        f (h5py.File)
    """
    
    # Create a group for the alphashapes
    alphashapes_group = f.create_group('alphashapes')
    
    # Loop over the cell_labels and save the alphashapes in the group
    for cellID in alphashapes:
        
        # Create a group for the cell
        cell_group = alphashapes_group.create_group(cellID)
        
        # Add the alpha attribute (float)
        cell_group.attrs['alpha'] = alphashapes[cellID]['alpha']
        
        # Save the volume of the mesh as an attribute
        cell_group.attrs['volume'] = alphashapes[cellID]['mesh'].volume
        
        # Save the area of the mesh as an attribute
        cell_group.attrs['area'] = alphashapes[cellID]['mesh'].area
        
        # Save the mesh vertices and faces as datasets
        cell_group.create_dataset('vertices', data=alphashapes[cellID]['mesh'].vertices)
        cell_group.create_dataset('faces', data=alphashapes[cellID]['mesh'].faces)

def load_cell_alphashape_from_hdf5(cellID: str, f: h5py.File) -> dict:
    """ Load the alphashape of a cell from an hdf5 file.

    Args:
        cellID (str)
        f (h5py.File)

    Returns:
        dict: alphashape[cellID] = {'alpha': float, 'mesh': trimesh.Trimesh}.
    """
    
    # Load the alpha value
    alpha = f['alphashapes'][cellID].attrs['alpha']
    
    # Load the volume and area of the mesh
    volume = f['alphashapes'][cellID].attrs['volume']
    area = f['alphashapes'][cellID].attrs['area']
    
    # Load the mesh vertices and faces
    vertices = f['alphashapes'][cellID]['vertices'][:]
    faces = f['alphashapes'][cellID]['faces'][:]
    
    # Create the mesh
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    
    # Assert that the volume and area of the mesh are correct
    assert np.isclose(mesh.volume, volume), 'The volume of the mesh is incorrect.'
    assert np.isclose(mesh.area, area), 'The area of the mesh is incorrect.'
    
    # Return the alphashape as a dictionary
    return {'alpha': alpha, 'mesh': mesh}
    
