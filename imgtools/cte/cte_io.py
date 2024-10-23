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

def add_key_to_attrs_in_hdf5(key: str, value, f: h5py.File) -> None:
    """ Add a key to the attributes in an hdf5 file. """
    f.attrs[key] = value


# SAVE/LOAD CELL LABELS

def save_cell_labels_to_hdf5(cell_labels: list, f: h5py.File) -> None:
    """ Save the cell_labels array to an hdf5 file.
    The cell_labels are saved as an array of 'S', whose length is the maximum length of the cellIDs. """
    f.create_dataset('cell_labels', data=np.array(cell_labels).astype('S'))

def load_cell_labels_from_hdf5(f: h5py.File) -> np.ndarray:
    """ Load the cell_labels array from an hdf5 file.
    The cell_labels are loaded as an array of str, i.e. 'U' (unicode string)."""
    cell_labels = f['cell_labels'][:].astype(str)
    return cell_labels

def pop_cell_labels_from_hdf5(f: h5py.File, cells_to_pop: list) -> None:
    """ Remove the cellIDs from the cell_labels array in the hdf5 file.

    Args:
        f (h5py.File)
        cells_to_pop (list): list of cellIDs to remove from the cell_labels array.
    """
    
    # Check if the cell_labels array exists in the hdf5 file
    if 'cell_labels' not in f:
        return None
    cell_labels = load_cell_labels_from_hdf5(f)
    
    # Remove the cellIDs from the cell_labels array
    mask = np.isin(cell_labels, cells_to_pop, invert=True)  # True for the cellIDs to keep
    cell_labels = cell_labels[mask]  # cell_labels subset with the cellIDs to keep
    
    # Remove the cell_labels array from the hdf5 file and save the new cell_labels array
    del f['cell_labels']
    save_cell_labels_to_hdf5(cell_labels, f)


# SAVE/LOAD CELL STATES

def save_cell_states_to_hdf5(cell_states: list, f: h5py.File) -> None:
    """ Save the cell_states array to an hdf5 file.
    The cell_states are saved as an array of 'S', whose length is the maximum length of the cell states. """
    f.create_dataset('cell_states', data=np.array(cell_states).astype('S'))

def load_cell_states_from_hdf5(f: h5py.File) -> np.ndarray:
    """ Load the cell_states array from an hdf5 file.
    The cell_states are loaded as an array of str, i.e. 'U' (unicode string)."""
    cell_states = f['cell_states'][:].astype(str)
    return cell_states

def pop_cell_states_from_hdf5(f: h5py.File, cells_to_pop: list) -> None:
    """ Remove the cellIDs from the cell_states array in the hdf5 file.

    Args:
        f (h5py.File)
        cells_to_pop (list): list of cellIDs to remove from the cell_states array.
    """
    
    # Check if the cell_states array exists in the hdf5 file
    if 'cell_states' not in f:
        return None
    cell_states = load_cell_states_from_hdf5(f)
    
    # Load the cell_labels array
    if 'cell_labels' not in f:
        raise ValueError('The cell_states array exists but the cell_labels array does not exist in the hdf5 file.')
    cell_labels = load_cell_labels_from_hdf5(f)
    
    # Remove the cellIDs from the cell_states array
    mask = np.isin(cell_labels, cells_to_pop, invert=True)  # True for the cellIDs to keep
    cell_states = cell_states[mask]  # cell_states subset with the cellIDs to keep
    
    # Remove the cell_states array from the hdf5 file and save the new cell_states array
    del f['cell_states']
    save_cell_states_to_hdf5(cell_states, f)


# SAVE/LOAD TRIAD LABELS

def save_triad_labels_to_hdf5(triad_labels: np.ndarray, f: h5py.File) -> None:
    """ Save the triad_labels array to an hdf5 file.
    Each triadID consists of cellID / chrom / traceID.
    It is an array of shape (ntriads, 3)."""
    f.create_dataset('triad_labels', data=np.array(triad_labels).astype('S'))

def load_triad_labels_from_hdf5(f: h5py.File) -> np.ndarray:
    """ Load the triad_labels array from an hdf5 file.
    Each triadID consists of cellID / chrom / traceID.
    It is an array of shape (ntriads, 3)."""
    triad_labels = f['triad_labels'][:].astype(str)
    return triad_labels


# SAVE/LOAD DATA

def save_cell_data_to_hdf5(cellID: str, cell_data: dict, f: h5py.File) -> None:
    """ Save the CTE data of a cell to an hdf5 file.
    
    The data is saved in numpy format in the group 'data', with a subgroup for the cellID.
    
    The subgroup contains the following datasets:
        'xs', 'ys', 'zs', 'chroms', 'starts', 'ends', 'lums', 'traceIDs', 'spotIDs'.

    Args:
        cellID (str)
        cell_data (dict): dictionary with the data.
        f (h5py.File): hdf5 file.
    """
    # Get the data group, create it if it does not exist
    data_group = f.require_group('data')
    # Create a group for the cell data
    cell_group = data_group.create_group(cellID)
    # Convert the cell data from dictionary to numpy format
    xs, ys, zs, chroms, starts, ends, lums, traceIDs, spotIDs = cte_utils.cell_dict_to_numpy(cell_data)
    # Save the cell data in the group
    cell_group.create_dataset('xs', data=xs)
    cell_group.create_dataset('ys', data=ys)
    cell_group.create_dataset('zs', data=zs)
    cell_group.create_dataset('chroms', data=chroms.astype('S'))
    cell_group.create_dataset('starts', data=starts)
    cell_group.create_dataset('ends', data=ends)
    cell_group.create_dataset('lums', data=lums)
    cell_group.create_dataset('traceIDs', data=traceIDs.astype('S'))
    cell_group.create_dataset('spotIDs', data=spotIDs.astype('S'))

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
    f.create_group('data')
    # Loop over the cell_labels and save the data in the group
    for cellID in data:
        save_cell_data_to_hdf5(cellID, data[cellID], f)

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
    chroms = cell_group['chroms'][:].astype(str)
    starts = cell_group['starts'][:]
    ends = cell_group['ends'][:]
    lums = cell_group['lums'][:]
    traceIDs = cell_group['traceIDs'][:].astype(str)
    spotIDs = cell_group['spotIDs'][:].astype(str)
    
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
    xs, ys, zs, chroms, starts, ends, lums, traceIDs, spotIDs = load_cell_data_from_hdf5(cellID, f, format='numpy')
    
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
        data = cte_utils.trace_numpy_to_dict(chrom, xs, ys, zs, starts, ends, lums, spotIDs)
    else:
        data = (xs, ys, zs, starts, ends, lums, spotIDs)
    
    return data

def pop_cell_data_from_hdf5(f: h5py.File, cells_to_pop: list) -> None:
    """ Remove the cellIDs from the data group in the hdf5 file.

    Args:
        f (h5py.File)
        cells_to_pop (list): list of cellIDs to remove from the data group.
    """
    
    # Check if the data group exists in the hdf5 file
    if 'data' not in f:
        return None
    
    # Loop over the cellIDs to pop and remove the cell data from the hdf5 file
    for cellID in cells_to_pop:
        if cellID in f['data']:
            del f['data'][cellID]

def pop_spot_data_from_hdf5(f: h5py.File, spots_to_pop: dict) -> int:
    """ Remove spotIDs from the data group in the hdf5 file.
    
    The data is loaded, modified and saved back to the hdf5 file.
    
    Also returns the number of removed spots.

    Args:
        f (h5py.File)
        spots_to_pop (dict): spotIDs to remove, in the format:
                             spots_to_pop[cellID][chrom][traceID] = [spotID1, spotID2, ...]
    
    Returns:
        (int): number of removed spots.
    """
    
    # Check if the data group exists in the hdf5 file
    if 'data' not in f:
        return 0
    
    # Initialize the counter of removed spots
    nspot_popped = 0
    
    # Loop over the cellIDs and remove the spotIDs from the data group
    for cellID in spots_to_pop:
        
        # Check if the cellID exists in the data group
        if cellID not in f['data']:
            continue
        
        # Load the cell data
        cell_data = load_cell_data_from_hdf5(cellID, f, format='dict')
        
        # Loop over the chrom/traceID/spotIDs to pop
        for chrom in spots_to_pop[cellID]:
            for traceID in spots_to_pop[cellID][chrom]:
                for spotID in spots_to_pop[cellID][chrom][traceID]:
                    
                    # Check if the chrom/traceID exists in the cell data
                    try:
                        cell_data[chrom][traceID][spotID]
                    except KeyError:
                        continue
                    
                    # Remove the spotID from the cell data
                    nspot_popped += 1
                    del cell_data[chrom][traceID][spotID]
        
        # Remove the cell data from the hdf5 file and save the new cell data
        del f['data'][cellID]
        save_cell_data_to_hdf5(cellID, cell_data, f)
        
    return nspot_popped

# SAVE/LOAD ALPHASHAPES

def save_cell_alphashape_to_hdf5(cellID: str, cell_alphashape: dict, f: h5py.File) -> None:
    """ Save the alphashape of a cell to an hdf5 file.

    Args:
        cellID (str)
        alphashape (dict): dictionary with the alphashape.
                            alphashape = {'alpha': float, 'mesh': trimesh.Trimesh}.
        f (h5py.File)
    """
    # Get the alphashapes group, create it if it does not exist
    alphashapes_group = f.require_group('alphashapes')
    # Create a group for the cell
    cell_group = alphashapes_group.create_group(cellID)
    # Add the alpha attribute (float)
    cell_group.attrs['alpha'] = cell_alphashape['alpha']
    # Save the volume of the mesh as an attribute
    cell_group.attrs['volume'] = cell_alphashape['mesh'].volume
    # Save the area of the mesh as an attribute
    cell_group.attrs['area'] = cell_alphashape['mesh'].area
    # Save the mesh vertices and faces as datasets
    cell_group.create_dataset('vertices', data=cell_alphashape['mesh'].vertices)
    cell_group.create_dataset('faces', data=cell_alphashape['mesh'].faces)

def save_alphashapes_to_hdf5(alphashapes: dict, f: h5py.File) -> None:
    """ Save the alphashapes to an hdf5 file.

    Args:
        alphashapes (dict): dictionary with the alphashapes.
                            alphashapes[cellID] = {'alpha': float, 'mesh': trimesh.Trimesh}.
        f (h5py.File)
    """
    
    # Create a group for the alphashapes
    f.create_group('alphashapes')
    # Loop over the cell_labels and save the alphashapes in the group
    for cellID in alphashapes:
        save_cell_alphashape_to_hdf5(cellID, alphashapes[cellID], f)

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

def pop_cell_alphashape_from_hdf5(f: h5py.File, cells_to_pop: list) -> None:
    """ Remove the cellIDs from the alphashapes group in the hdf5 file.

    Args:
        f (h5py.File)
        cells_to_pop (list): list of cellIDs to remove from the alphashapes group.
    """
    
    # Check if the alphashapes group exists in the hdf5 file
    if 'alphashapes' not in f:
        return None
    
    # Loop over the cellIDs to pop and remove the cell alphashapes from the hdf5 file
    for cellID in cells_to_pop:
        if cellID in f['alphashapes']:
            del f['alphashapes'][cellID]    


# MERGE FUNCTION

def merge_group_from_hdf5(
    group: str, f1: h5py.File, f2: h5py.File, f12: h5py.File,
    tag1: str = None, tag2: str = None
) -> None:
    """ Merge the data of a group - containing subgroups for each cell - from two hdf5 files into a third hdf5 file.
    
    The cellIDs from the first file are extended with the tag1, and same for the second file with the tag2.
    
    Can be used for both the 'data' and 'alphashapes' groups.

    Args:
        group (str): name of the group to merge.
        f1 (h5py.File): first hdf5 file to merge.
        f2 (h5py.File): second hdf5 file to merge.
        f12 (h5py.File): merged hdf5 file.
        tag1 (str, optional): tag to add to the cellIDs from the first file. Default: None.
        tag2 (str, optional): tag to add to the cellIDs from the second file. Default: None.
    """
    
    # Check that the data group exists in the hdf5 files
    if group not in f1:
        raise ValueError(f'The group {group} does not exist in the first hdf5 file.')
    if group not in f2:
        raise ValueError(f'The group {group} does not exist in the second hdf5 file.')
    
    # Check that the data group does NOT exist in the merged hdf5 file
    if group in f12:
        raise ValueError(f'The group {group} already exists in the merged hdf5 file.')
    
    # Create the group in the merged hdf5 file
    f12.create_group(group)
    
    # Loop over the cellIDs in the first hdf5 file and copy the cell subgroup to the merged hdf5 file (adding tag1 to the cellID)
    for cellID in f1['data']:
        cellID_new = f'{cellID}_{tag1}' if tag1 is not None else cellID
        f1.copy(f'{group}/{cellID}', f12, name=f'{group}/{cellID_new}')
    
    # Loop over the cellIDs in the second hdf5 file and copy the cell data to the merged hdf5 file (adding tag2 to the cellID)
    for cellID in f2['data']:
        cellID_new = f'{cellID}_{tag2}' if tag2 is not None else cellID
        f2.copy(f'{group}/{cellID}', f12, name=f'{group}/{cellID_new}')



# CONSISTENCY CHECK FUNCTION

def check_consistency(f: h5py.File) -> None:
    """ Check the consistency of the hdf5 file of a ChromatinTracingExperiment.
    
    Makes sure that the datasets and groups in the hdf5 file are consistent with each other.

    Args:
        f (h5py.File)
    """
    
    # Check that the core groups and datasets exist in the hdf5 file: index, cell_labels, data
    if 'index' not in f:
        raise ValueError('The index does not exist in the hdf5 file.')
    if 'cell_labels' not in f:
        raise ValueError('The cell_labels array does not exist in the hdf5 file.')
    if 'data' not in f:
        raise ValueError('The data group does not exist in the hdf5 file.')
    
    # Get the cell_labels array
    cell_labels = load_cell_labels_from_hdf5(f)
    
    # Check that the cell_labels array has the same length as the data group
    if len(cell_labels) != len(f['data']):
        raise ValueError('The cell_labels array and the data group have different lengths.')
    
    # Check that all the cellIDs from the cell_labels array are in the data group
    for cellID in cell_labels:
        if cellID not in f['data']:
            raise ValueError(f'The cellID {cellID} from the cell_labels array is not in the data group.')
    
    # Viceversa, check that all the cellIDs in the data group are in the cell_labels array
    for cellID in f['data']:
        if cellID not in cell_labels:
            raise ValueError(f'The cellID {cellID} from the data group is not in the cell_labels array.')
    
    # If the cell_states array exists, check it
    if 'cell_states' in f:
        cell_states = load_cell_states_from_hdf5(f)
        # Check that the cell_states array has the same length as the cell_labels array
        if len(cell_states) != len(cell_labels):
            raise ValueError('The cell_states array and the cell_labels array have different lengths.')
    
    # If the alphashapes group exists, check it
    if 'alphashapes' in f:
        # Check that the alphashapes group has the same length as the cell_labels array
        if len(cell_labels) != len(f['alphashapes']):
            raise ValueError('The cell_labels array and the alphashapes group have different lengths.')
        # Check that all the cellIDs from the cell_labels array are in the alphashapes group
        for cellID in cell_labels:
            if cellID not in f['alphashapes']:
                raise ValueError(f'The cellID {cellID} from the cell_labels array is not in the alphashapes group.')
        # Viceversa, check that all the cellIDs in the alphashapes group are in the cell_labels array
        for cellID in f['alphashapes']:
            if cellID not in cell_labels:
                raise ValueError(f'The cellID {cellID} from the alphashapes group is not in the cell_labels array.')
