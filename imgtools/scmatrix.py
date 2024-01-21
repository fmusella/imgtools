import os
import numpy as np
import pickle
from alabtools.utils import Index


class SingleCellMatrix:
    """ A class to store counts data from single-cell DNA experiments,
    where the data are organized as a matrix of shape ncells x ndomains x ncopies.
    
    The data structure describes the chromosomal domains with the Index object,
    and the cells with a cell label (e.g. cell ID) and a cell state (e.g. cell cycle phase).
    
    Finally, the class also stores a dictionary to track spot IDs to their index in the matrix.
    
    --------------------
    Attributes:
        index (Index): Index object.
        cell_labels (np.ndarray, dtype='U10'): cell labels.
        cell_states (np.ndarray, dtype='U10'): cell states.
        matrix (np.ndarray, dtype='int32' or 'float32'): ncells x ndomains x ncopies matrix.
        spot_hash (dict): dictionary to track spot IDs to their index in the matrix.
    
    --------------------
    """
    
    def __init__(self) -> None:
        self.index = None
        self.cell_lables = None
        self.cell_states = None
        self.matrix = None
        self.spot_hash = None
    
    
    # INPUT/OUTPUT FUNCTIONS
    
    def save(self, filename: str) -> None:
        """Saves the object to a pickle file.

        Args:
            filename (str): name of the directory where the object will be saved.

        Raises:
            TypeError: filename is not a string.
            FileNotFoundError: filename is not a valid directory.
        """

        # Check that filename is a string and that the directory exists
        if not isinstance(filename, str):
            raise TypeError("filename must be a string.")
        if not os.path.exists(os.path.dirname(filename)):
            raise NotADirectoryError("Directory {} does not exist.".format(os.path.dirname(filename)))

        # Save the object to a pickle file
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    def load(self, filename: str) -> None:
        """Loads a SingleCellMatrix object from a pickle file.

        Args:
            filename (str): path and name of the pickle file.

        Raises:
            TypeError: filename is not a string.
            FileNotFoundError: filename is not a valid file.
            Exception: the object could not be loaded from the file.
            TypeError: the loaded object is not a SingleCellMatrix object.
            Exception: the loaded object does not have data.
        """

        # Check that filename is a string and that the file exists
        if not isinstance(filename, str):
            raise TypeError("filename must be a string.")
        if not os.path.exists(filename):
            raise FileNotFoundError("File {} does not exist.".format(filename))

        # Try to load the object from the pickle file
        try:
            with open(filename, 'rb') as f:
                loaded_object = pickle.load(f)
        except:
            raise Exception("Could not load object from file {}.".format(filename))

        # Check that the loaded object is a SingleCellMatrix object and that it has data
        if not isinstance(loaded_object, SingleCellMatrix):
            raise TypeError("Loaded object is not a SingleCellMatrix object.")
        if loaded_object.matrix is None:
            raise Exception("Loaded object does not have data.")

        # Update the attributes of the current SingleCellMatrix object
        self.__dict__.update(loaded_object.__dict__)
        
        del loaded_object
    
    def add_data(self,
                 index: Index,
                 cell_labels: np.ndarray,
                 matrix: np.ndarray,
                 spot_hash: dict,
                 cell_states: np.ndarray = None) -> None:
        """ Add data to the SingleCellMatrix object.

        Args:
            index (Index): Index object.
            cell_labels (np.ndarray, dtype='U10'): cell labels.
            matrix (np.ndarray, dtype='int32' or 'float32'): ncells x ndomains x ncopies matrix.
            spot_hash (dict): dictionary to track spot IDs to their index in the matrix.
            cell_states (np.ndarray, dtype='U10', optional): cell states. Defaults to None.
        """
        
        # Check that the Index object is valid
        if not isinstance(index, Index):
            raise TypeError("index must be an Index object.")
        
        # Check that the cell labels are valid
        if not isinstance(cell_labels, np.ndarray):
            raise TypeError("cell_labels must be a numpy array.")
        if cell_labels.dtype != 'U10':
            raise TypeError("cell_labels must be a numpy array of strings.")
        
        # Check that the matrix is valid
        if not isinstance(matrix, np.ndarray):
            raise TypeError("matrix must be a numpy array.")
        if matrix.dtype not in ['int32', 'float32']:
            raise TypeError("matrix must be a numpy array of integers or floats.")
        if matrix.ndim != 3:
            raise TypeError("matrix must be a 3-dimensional numpy array.")
        if matrix.shape[0] != len(cell_labels):
            raise TypeError("matrix must have the same number of cells as cell_labels.")
        if matrix.shape[1] != len(index):
            raise TypeError("matrix must have the same number of domains as index.")
        
        # Check that the spot hash is valid
        if not isinstance(spot_hash, dict):
            raise TypeError("spot_hash must be a dictionary.")
        
        # Check that the cell states are valid
        if cell_states is not None:
            if not isinstance(cell_states, np.ndarray):
                raise TypeError("cell_states must be a numpy array.")
            if cell_states.dtype != 'U10':
                raise TypeError("cell_states must be a numpy array of strings.")
            if len(cell_states) != len(cell_labels):
                raise TypeError("cell_states must have the same number of cells as cell_labels.")
        
        # Update the attributes of the SingleCellMatrix object
        self.index = index
        self.cell_labels = cell_labels
        self.cell_states = cell_states
        self.matrix = matrix
        self.spot_hash = spot_hash
    
    
    # Next time I have to write a function that produces a SingleCellMatrix object in the CTE class
    
    # Functions I want to put next (for now they are place-holders)
    
    def compute_rt(self, isolate_state: str = None) -> np.ndarray:
        # Compute the Replication Timing (RT) of each domain across all cells (or the subset of cells with a given state)
        return None
    
    def haploid_sort_by_row(self, isolate_state: str = None, sorter: np.ndarray = None) -> (np.ndarray, np.ndarray):
        # Creates a haploid version of the matrix, with copies stacked on top of each other, sorted by ascending value in the sorter array
        return None


# I want to put run_cellcycle and run_replication outside the class
# The reason is that these functions are specific to a particular type of data (raw counts), and thus it's better to have them as separate functions
# Indeed, I can't use them for process normalized data, for example
