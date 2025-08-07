import numpy as np

def get_flattened_indices(ncells: int, nloci: int, ncopies: int) -> np.ndarray:
    
    cells_indices = np.arange(ncells)
    loci_indices = np.arange(nloci)
    copies_indices = np.arange(ncopies)
    indices = []
    for cellnum in cells_indices:
        for locinum in loci_indices:
            for copynum in copies_indices:
                indices.append([cellnum, locinum, copynum])
    indices = np.array(indices)
    return indices

"""def get_flattened_indices(ncells: int, nloci: int, ncopies: int) -> np.ndarray:
    cells_indices = np.repeat(np.arange(ncells), nloci * ncopies)
    loci_indices = np.tile(np.repeat(np.arange(nloci), ncopies), ncells)
    copies_indices = np.tile(np.arange(ncopies), ncells * nloci)
    indices = np.column_stack((cells_indices, loci_indices, copies_indices))
    return indices"""