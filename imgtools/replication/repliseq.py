import os
import numpy as np
import h5py
from alabtools.utils import Index
from ..scf import SingleCellFeature


class SimulatedRepliSeqExperiment:
    
    # INITIALIZATION METHODS
    
    def __init__(self) -> None:
        """ Initialize the SimulatedRepliSeqExperiment object.
        
        This method just initializes the attributes as None.
        Then, either 'from_hdf5' or 'from_scf' should be called to initialize the object,
        either from a HDF5 file or from a SingleCellFeature object.
        """
        
        # Initialize as None the attributes that will be set later
        self.ncells = None
        self.nloci = None
        self.ncopies = None
        self.index = None
        self.states = None
        self.volumes = None
        self.n_ic = None
    
    @classmethod
    def from_hdf5(cls, filename: str) -> 'SimulatedRepliSeqExperiment':
        """ Initializes the SimulatedRepliSeqExperiment object by loading the data from an HDF5 file.
        
        Args:
            filename (str): name, with path, of the HDF5 file to load the data.
        
        Returns:
            SimulatedRepliSeqExperiment
        """
        
        # Create a new object
        obj = cls()
        # Load the data from the HDF5 file
        obj.load_from_hdf5(filename)
        return obj
    
    
    @classmethod
    def from_scf(cls, scf: SingleCellFeature) -> 'SimulatedRepliSeqExperiment':
        """ Initializes the SimulatedRepliSeqExperiment object from a SingleCellFeature object.
        
        Args:
            scf (SingleCellFeature)
        
        Returns:
            SimulatedRepliSeqExperiment
        """
        
        obj = cls()
        
        # Check the input SingleCellFeature object
        obj._check_scf(scf)
        
        obj.index = scf.index
        obj.states = scf.cell_states
        obj.volumes = scf.volumes
        obj.n_ic = scf.get_feature('spotcount')
        obj.ncells, obj.nloci, obj.ncopies = obj.n_ic.shape
        
        return obj
    
    @staticmethod
    def _check_scf(scf: SingleCellFeature) -> None:
        """ Check the input SingleCellFeature object.
        
        It checks that:
         - the input is a SingleCellFeature object,
         - the SCF contains the 'spotcount' feature,
         - the SCF contains the 'cell_states' feature,
         - the 'cell_states' feature only contains 'G1', 'S' and 'G2',
         - the index of the SCF has a valid resolution with consecutive loci.

        Args:
            scf (SingleCellFeature)
        """
        
        if not isinstance(scf, SingleCellFeature):
            raise TypeError("The input scf must be a SingleCellFeature.")
        
        if 'spotcount' not in scf.feature_list:
            raise ValueError("The input scf must contain the 'spotcount' feature.")
        if 'cell_states' not in scf:
            raise ValueError("The input scf must contain the 'cell_states' dataset.")
        if not all([state in ['G1', 'S', 'G2'] for state in scf.cell_states]):
            raise ValueError("The 'cell_states' feature must only contain 'G1', 'S' and 'G2'.")
        if 'volumes' not in scf:
            raise ValueError("The input scf must contain the 'volumes' dataset.")
        
        if scf.index.resolution() is None:
            raise ValueError("The index of the input SCF must have a valid resolution.")
        if not scf.index.consecutive():
            raise ValueError("The index of the input SCF must have consecutive loci.")
    
    
    # INPUT/OUTPUT METHODS
    
    def save_to_hdf5(self, filename: str) -> None:
        """ Save the data of the object to an HDF5 file.
        
        To identify the data to store, it uses the keys of the object's __dict__ attribute.
        It doesn't store a few keys that are not relevant to the analysis.
        
        Args:
            filename (str): name, with path, of the HDF5 file to save the data.
        """
        
        # Check that the filename has a valid path
        if not os.path.exists(os.path.dirname(filename)):
            raise ValueError(f"Invalid path: {filename}")
        # Check that the filename doesn't already exist
        if os.path.exists(filename):
            print(f"Warning: {filename} already exists. Can't overwrite it.")
            return
        
        # Create the HDF5 file and save the data
        with h5py.File(filename, 'w') as f:
            
            # Save the index
            self.index.save(f)
            
            # If the object has a config dictionary, save it as a group
            if hasattr(self, 'config'):
                config_group = f.create_group('config')
                for key, value in self.config.items():
                    config_group.attrs[key] = value
            
            # Loop over the items of the object to save arrays as datasets
            for key, value in self.__dict__.items():

                # Ignore the keys that are saved in a different way
                keys_to_ignore = ['config', 'genome', 'index', 'ncells', 'nloci', 'ncopies']
                if key in keys_to_ignore:
                    continue
                # Ignore the keys that are not numpy arrays
                if not isinstance(value, np.ndarray):
                    continue
                # If the array is a string, save as S type
                if value.dtype.kind in ['U', 'S']:
                    f.create_dataset(key, data=value.astype('S'))
                # Otherwise, save with the default type
                else:
                    f.create_dataset(key, data=value)
    
    def load_from_hdf5(self, filename: str) -> None:
        """ Load the data of the object from an HDF5 file.
        
        It loads the data from the HDF5 file to the object's attributes.
        It doesn't load a few keys that are not relevant to the analysis.
        
        Args:
            filename (str): name, with path, of the HDF5 file to load the data.
        """
        
        # Check that the filename exists
        if not os.path.exists(filename):
            raise ValueError(f"File not found: {filename}")
        
        # Load the data from the HDF5 file
        with h5py.File(filename, 'r') as f:
            
            # Loop over the items of the object and load the data
            for key in f.keys():
                
                # If the key is 'config', load as a dictionary
                if key == 'config':
                    self.config = {k: v for k, v in f[key].attrs.items()}
                    continue
                
                # If the key is 'genome', skip it (it is loaded in the Index object)
                if key == 'genome':
                    continue
                
                # If the key is 'index', load as an Index object
                if key == 'index':
                    self.index = Index(f)
                    continue
                
                # Otherwise, load as a numpy array
                arr = f[key][:]
                # If the array is a string, convert to unicode string
                if arr.dtype.kind in ['U', 'S']:
                    arr = arr.astype(str)
                # Store the array in the object
                self.__dict__[key] = arr
        
        # Set the number of cells, loci and copies as attributes
        self.ncells, self.nloci, self.ncopies = self.n_ic.shape