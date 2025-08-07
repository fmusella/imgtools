import os
import numpy as np
import h5py
from ..scf import SingleCellFeature
from ..scf import scf_utils
from .repliseq import SimulatedRepliSeqExperiment


class SimulatedSingleCellRepliSeqExperiment:
    """_summary_
    """
    
    def __init__(self, h5_name: str, mode: str, win_size: float):
        
        # Extend the name with its absolute path
        h5_name = os.path.abspath(h5_name)
        # Check that file has a valid path
        if not os.path.exists(os.path.dirname(h5_name)):
            raise FileNotFoundError("The path of the HDF5 file does not exist.")
        # Store the name of the HDF5 file
        self.h5_name = h5_name
        # Read / create the HDF5 file
        self.h5 = h5py.File(h5_name, mode=mode)
        
        # Store the window size
        self.WIN_SIZE = win_size
    
    def read_features(
        self,
        scf: SingleCellFeature,
        simrep: SimulatedRepliSeqExperiment,
        features: list
    ):
        
        # Get the shape of the SCF matrices
        ncells, nloci, ncopies = scf.get_expected_shape()
        self.ncells, self.nloci, self.ncopies = ncells, nloci, ncopies
        
        

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
