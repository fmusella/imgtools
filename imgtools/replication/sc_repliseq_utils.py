import numpy as np

def get_flattened_indices(ncells: int, nloci: int, ncopies: int) -> np.ndarray:
    """ Generate the indices to convert a flattened array (in 'C' order)
    back to its original shape: (ncells, nloci, ncopies).
    
    The indices will be in the format:
    [[cellnum_0, locinum_0, copy_0],
     [cellnum_0, locinum_0, copy_1],
     [cellnum_0, locinum_1, copy_0],
                ...,
     [cellnum_ncells-1, locinum_nloci-1, copy_ncopies-1]]

    Args:
        ncells (int)
        nloci (int)
        ncopies (int)

    Returns:
        np.ndarray: An array of shape (ncells * nloci * ncopies, 3)
    """
    cells_indices = np.repeat(np.arange(ncells), nloci * ncopies)
    loci_indices = np.tile(np.repeat(np.arange(nloci), ncopies), ncells)
    copies_indices = np.tile(np.arange(ncopies), ncells * nloci)
    indices = np.column_stack((cells_indices, loci_indices, copies_indices))
    return indices

def tile_to_shape(x: np.ndarray, ncells: int, nloci: int, ncopies: int) -> np.ndarray:
    """ Tile an array of shape (ncells,) or (nloci,) to shape (ncells, nloci, ncopies).
    
    The code checks how to tile based on the length of the input array.

    Args:
        x (np.ndarray): Array to be tiled, of shape (ncells,) or (nloci,).
        ncells (int)
        nloci (int)
        ncopies (int)

    Returns:
        np.ndarray: Tiled array of shape (ncells, nloci, ncopies).
    """
    
    # Check that x is a 1D array
    if not x.ndim == 1:
        raise ValueError("Input array x must be 1-dimensional.")
    
    if len(x) == ncells:
        x_tiled = np.tile(x[:, np.newaxis, np.newaxis], (1, nloci, ncopies))
        return x_tiled
    elif len(x) == nloci:
        x_tiled = np.tile(x[np.newaxis, :, np.newaxis], (ncells, 1, ncopies))
        return x_tiled
    else:
        raise ValueError(f"Input array x must have length {ncells} or {nloci}, but has length {len(x)}.")
