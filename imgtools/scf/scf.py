import os
import numpy as np
import h5py
from alabtools.utils import Index


class SingleCellFeature:
    """ A class to store feature data from single-cell DNA experiments, e.g. DNA counts,
    where the data are organized as a matrix of shape ncells x ndomains x ncopies.
    
    The data structure describes the chromosomal domains with the Index object,
    and the cells with a cell label (e.g. cell ID) and a cell state (e.g. cell cycle phase).
    
    ----------
    Attributes:
        h5_name (str): path and name of the HDF5 file.
        h5 (h5py.File): HDF5 file to store the data.
                        Contains the following groups:
                        - index: Index object.
                        - attrs: attributes.
                        - cell_labels: array with the cell IDs.
                        - cell_states: array with the cell states.
                        - volumes: array with the cell volumes.
                        - feature_list: list of feature matrices.
                        - [feature]: feature matrix. (saved with a particular name)
    ---------- 
    Properties (from h5 file):
        index (Index): Index object.
        attrs (dict): attributes.
        cell_labels (np.ndarray): array with the cell IDs.
        cell_states (np.ndarray): array with the cell states.
        volumes (np.ndarray): array with the cell volumes.
        feature_list (list): list of feature matrices.
    """
    
    def __init__(self, h5_name: str, mode: str = 'r') -> None:
        """ Initialize the SingleCellFeature object.
        
        A HDF5 file is created to store the data.
        
        The file is opened in the specified mode, that should match the use case,
        e.g. a file cannot be created if the mode is 'r'.

        Args:
            h5_name (str): path and name of the HDF5 file.
            mode (str): 'r', 'r+', 'w', 'w-', 'x', 'a'. Defaults to 'r'.
        """
        
        # Extend the name with its absolute path
        h5_name = os.path.abspath(h5_name)
        
        # Check that h5_name has a valid path
        if not os.path.exists(os.path.dirname(h5_name)):
            raise FileNotFoundError("The path of the HDF5 file does not exist.")
        
        # Check that mode is valid
        if not mode in ['r', 'r+', 'w', 'w-', 'x', 'a']:
            raise ValueError("mode must be one of 'r', 'r+', 'w', 'w-', 'x', 'a'.")
        
        # If the file doesn't exists, make sure that mode is write (w, w-, x, r+, a)
        if not os.path.exists(h5_name) and mode not in ['w', 'w-', 'x', 'r+', 'a']:
            raise FileNotFoundError("The HDF5 file does not exist. Use mode 'w', 'w-', 'x', 'r+', 'a' to create it.")
        
        # Open the HDF5 file
        self.h5_name = h5_name
        self.h5 = h5py.File(h5_name, mode)
    
    
    # CONTAIN METHOD
    def __contains__(self, name: str) -> bool:
        """ Check if a dataset exists in the h5 file."""
        return name in self.h5
    
    
    # SETTER FUNCTIONS
    
    def set_index(self, index: Index) -> None:
        """ Set the Index object in the h5 file."""
        index.save(self.h5)
    
    def set_attrs(self, attrs: dict) -> None:
        """ Save the attributes in the h5 file.
        Attributes are stored in the root of the h5 file."""
        for key in attrs:
            self.h5.attrs[key] = attrs[key]
    
    def set_cell_labels(self, cell_labels: np.ndarray) -> None:
        """ Save the cell labels in the h5 file.
        Cell labels are string, so they must be converted to 'S' type.
        We use a length of 20 to be sure that the strings are not truncated."""
        self.h5.create_dataset('cell_labels', data=np.array(cell_labels).astype('S20'), dtype=np.dtype('S20'))
    
    def set_cell_states(self, cell_states: np.ndarray) -> None:
        """ Save the cell states in the h5 file.
        Cell states are string, so they must be converted to 'S' type.
        We use a length of 20 to be sure that the strings are not truncated."""
        self.h5.create_dataset('cell_states', data=np.array(cell_states).astype('S20'), dtype=np.dtype('S20'))
    
    def set_volumes(self, volumes: np.ndarray) -> None:
        """ Save the cell volumes in the h5 file."""
        self.h5.create_dataset('volumes', data=volumes, dtype='float32')
    
    def set_matrix(self, matrix: np.ndarray, name: str) -> None:
        """ Save the feature matrix in the h5 file."""
        # Check that the matrix is not already in the h5 file
        if name in self:
            raise ValueError("The feature matrix '{}' already exists in the h5 file.".format(name))
        # Add the matrix to the h5 file
        self.h5.create_dataset(name, data=matrix, dtype=matrix.dtype)
        
    
    
    # GETTER FUNCTIONS
    
    def get_index(self) -> Index:
        """ Get the Index object from the h5 file."""
        return Index(self.h5)
    
    def get_attrs(self) -> dict:
        """ Get the attributes from the h5 file."""
        attrs = {}
        for key in self.h5.attrs:
            attrs[key] = self.h5.attrs[key]
        return attrs
    
    def get_cell_labels(self) -> np.ndarray:
        """ Get the cell labels from the h5 file.
        Cell labels are string, we retrieve them in 'U' type."""
        return self.h5['cell_labels'][:].astype('U20')
    
    def get_cellnum(self, cellID: str) -> int:
        """ Get the cell number of the input cellID.
        The number is the index of cellID in the cell_labels array."""
        cell_labels = self.get_cell_labels()
        return np.where(cell_labels == cellID)[0][0]
    
    def get_cellID(self, cellnum: int) -> str:
        """ Get the cellID of the input cell number.
        The cellID is the value of cell_labels at index cellnum."""
        cell_labels = self.get_cell_labels()
        return cell_labels[cellnum]
    
    def get_cell_states(self) -> np.ndarray:
        """ Get the cell states from the h5 file.
        Cell states are string, we retrieve them in 'U' type."""
        return self.h5['cell_states'][:].astype('U20')
    
    def get_volumes(self) -> np.ndarray:
        """ Get the cell volumes from the h5 file."""
        return self.h5['volumes'][:]
    
    def get_matrix(self, name: str, cellID: str = None) -> np.ndarray:
        """ Get the feature matrix from the h5 file.
        The feature matrix is a 3D array of shape ncells x ndomains x ncopies.
        It can be retrieved for all cells or for a specific cellID."""
        if cellID is None:
            return self.h5[name][:]
        else:
            cellnum = self.get_cellnum(cellID)
            return self.h5[name][cellnum, :, :]
    
    def get_feature_list(self) -> list:
        """ Get the list of feature matrices in the h5 file."""
        # Get the list of keys in the h5 file
        h5_keys = list(self.h5.keys())
        # Remove the keys that are not feature matrices
        remove_keys = ['index', 'genome', 'cell_labels', 'cell_states', 'volumes']
        for key in remove_keys:
            if key in h5_keys:
                h5_keys.remove(key)
        return h5_keys
    
    
    # DEFINE PROPERTIES (READ ONLY)
    index = property(get_index, doc="Index object.")
    attrs = property(get_attrs, doc="Attributes.")
    cell_labels = property(get_cell_labels, doc="Cell labels.")
    cell_states = property(get_cell_states, doc="Cell states.")
    volumes = property(get_volumes, doc="Cell volumes.")
    feature_list = property(get_feature_list, doc="List of feature matrices.")
    
    
    # INPUT/OUTPUT FUNCTIONS
    
    def close(self) -> None:
        """ Close the HDF5 file."""
        self.h5.close()
    
    def add_index_attrs_cell_labels(self, index: Index, attrs: dict, cell_labels: np.ndarray) -> None:
        """ Add the Index object, the attributes, and the cell labels to the h5 file, checking consistency.

        Args:
            index (Index)
            attrs (dict): attributes of the data.
            cell_labels (np.ndarray, str): array with the cell IDs.
        """
        
        # Check that the Index object is valid
        if not isinstance(index, Index):
            raise TypeError("index must be an Index object.")
        
        # Check that the attributes are valid
        if not isinstance(attrs, dict):
            raise TypeError("attrs must be a dictionary.")
        required_attrs_keys = ['ncell', 'max_ntrace_per_chrom']
        for key in required_attrs_keys:
            if key not in attrs:
                raise ValueError("attrs must have the key '{}'.".format(key))
        
        # Check that the cell labels are valid
        if not isinstance(cell_labels, np.ndarray):
            raise TypeError("cell_labels must be a numpy array.")
        if not issubclass(cell_labels.dtype.type, np.str_):
            raise TypeError("cell_labels must be a numpy array of strings.")
        if not len(cell_labels) == attrs['ncell']:
            raise ValueError("cell_labels must have the same number of cells as ncell in attrs.")
        
        self.set_index(index)
        self.set_attrs(attrs)
        self.set_cell_labels(cell_labels)
    
    def add_matrix(self, matrix: np.ndarray, name: str) -> None:
        """ Add a feature matrix to the h5 file, checking consistency.

        Args:
            matrix (np.ndarray, int/float): matrix of shape ncells x ndomains x ncopies.
            name (str): name of the feature associated to the matrix.
        """
        
        # Check that the name is valid
        if not isinstance(name, str):
            raise TypeError("The name of the matrix must be a string.")
        
        # Check that the matrix is a numpy array
        if not isinstance(matrix, np.ndarray):
            raise TypeError("The matrix must be a numpy array.")
        # Check that the matrix is either int or float
        if not np.issubdtype(matrix.dtype, np.integer) and not np.issubdtype(matrix.dtype, np.floating):
            raise TypeError("The matrix must be a numpy array of integers or floats.")
        # Check that the matrix has the right shape
        shape_expected = (self.attrs['ncell'], len(self.index), self.attrs['max_ntrace_per_chrom'])
        if not matrix.shape == shape_expected:
            raise ValueError("The shape of the matrix is not valid. Expected: {}, Found: {}.".format(shape_expected, matrix.shape))
        
        self.set_matrix(matrix, name)
    
    def add_volumes(self, volumes: np.ndarray) -> None:
        """ Add the cell volumes to the h5 file, checking consistency.

        Args:
            volumes (np.ndarray): numpy array of cell volumes of shape (ncells,)
        """
        
        # Check that volumes is a numpy array
        if not isinstance(volumes, np.ndarray):
            raise TypeError("volumes must be a numpy array.")
        # Check that volumes is a float array
        if not np.issubdtype(volumes.dtype, np.floating):
            raise TypeError("volumes must be a numpy array of floats.")
        # Check that volumes has the right shape (ncells,)
        if len(volumes) != len(self.cell_labels):
            raise TypeError("volumes must have the same number of cells as cell_labels.")
        
        self.set_volumes(volumes)
    
    def add_cell_states(self, cell_states: np.ndarray) -> None:
        """ Add the cell states to the h5 file, checking consistency.

        Args:
            cell_states (np.ndarray): numpy array of cell states of shape (ncells,)
        """
        
        # Check that cell_states is a numpy array
        if not isinstance(cell_states, np.ndarray):
            raise TypeError("volumes must be a numpy array.")
        # Check that cell_states is a string array
        if not np.issubdtype(cell_states.dtype, np.str_):
            raise TypeError("cell_states must be a numpy array of strings.")
        # Check that cell_states has the right shape (ncells,)
        if len(cell_states) != len(self.cell_labels):
            raise TypeError("cell_states must have the same number of cells as cell_labels.")
        
        self.set_cell_states(cell_states)
    
    
    # COMPUTATION FUNCTIONS
    
    def haploid_profile(self, feature_name: str, isolate_state: str = None, norm_by_volume: bool = False, zscore: bool = False) -> (np.ndarray, np.ndarray):
        """ Computes a 1D haploid profile of the required feature matrix, providing the mean and the standard deviation.
        If isolate_state is provided, it is computed only for the cells in that state (e.g. S phase).
        The feature matrix can be normalized by the cell volume and/or z-scored (if both are True, the feature matrix is first normalized by the cell volume and then z-scored).

        Args:
            feature_name (str): name of the feature matrix to compute the profile.
            isolate_state (str, optional): cell state to isolate. Defaults to None.
            norm_by_volume (bool, optional): if True, the feature matrix is normalized by the cell effective radius. Defaults to False.
            zscore (bool, optional): if True, the feature matrix is z-scored. Defaults to False.

        Returns:
            mean (np.ndarray): 1D haploid profile of the data.
            std (np.ndarray): 1D haploid standard deviation of the data.
        """
        
        # Get the feature matrix
        mat = self.get_matrix(feature_name)
        
        # If norm_by_vol is True, the feature matrix is normalized by the cell effective radius
        if norm_by_volume:
            vol = self.volumes
            rad = (3 * vol / (4 * np.pi))**(1/3)
            mat = mat / rad[:, np.newaxis, np.newaxis]
        
        # If zscore is True, the feature matrix is z-scored (each cell is z-scored independently)
        if zscore:
            mean = np.nanmean(mat, axis=(1, 2))[:, np.newaxis, np.newaxis]
            std = np.nanstd(mat, axis=(1, 2))[:, np.newaxis, np.newaxis]
            mat = (mat - mean) / std
        
        # Select only cells in the specified state if isolate_state is provided
        if isolate_state is not None:
            if not 'cell_states' in self:
                raise ValueError("Cell states are not defined. Cannot isolate state.")
            if not isolate_state in self.cell_states:
                raise ValueError("State {} is not defined. Cannot isolate state.".format(isolate_state))
            mask = self.cell_states == isolate_state
        # Otherwise, select all cells
        else:
            mask = np.ones(len(self.cell_labels), dtype=bool)
            
        # Compute the mean and standard deviation
        mean = np.nanmean(mat[mask, :, :], axis=(0, 2))  # np.array of shape (ndomains,)
        std = np.nanstd(mat[mask, :, :], axis=(0, 2))
        
        return mean, std
    
    def haploid_sort_by_row(self, isolate_state: str = None, sorter: np.ndarray = None) -> (np.ndarray, np.ndarray):
        # Placeholder
        # Creates a haploid version of the matrix, with copies stacked on top of each other, sorted by ascending value in the sorter array
        return None
    