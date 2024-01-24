import os
import sys
import numpy as np
import h5py
import pickle
import tempfile
from functools import partial
from alabtools.utils import Genome, Index
from alabtools.parallel import Controller
from . import cellcycle


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
        volumes (np.ndarray, dtype='float32'): cell volumes.
        matrix (np.ndarray, dtype='int32' or 'float32'): ncells x ndomains x ncopies matrix.
        spot_hash (dict): dictionary to track spot IDs to their index in the matrix.
    
    --------------------
    """
    
    def __init__(self) -> None:
        self.index = None
        self.cell_lables = None
        self.cell_states = None
        self.volumes = None
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
                 volumes: np.ndarray = None,
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
        
        # Check that the cell volumes are valid
        if volumes is not None:
            if not isinstance(volumes, np.ndarray):
                raise TypeError("volumes must be a numpy array.")
            if volumes.dtype != 'float32':
                raise TypeError("volumes must be a numpy array of floats.")
            if len(volumes) != len(cell_labels):
                raise TypeError("volumes must have the same number of cells as cell_labels.")
        
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
        self.volumes = volumes
        self.matrix = matrix
        self.spot_hash = spot_hash
    
    
    # Manipolation and data retrieval functions
    
    def get_profile(self, isolate_state: str = None) -> np.ndarray:
        """ Computes a 1D haploid profile of the data.
        
        If isolate_state is provided, it is computed only for the cells in that state (e.g. S phase)

        Args:
            isolate_state (str, optional): cell state to isolate. Defaults to None.

        Returns:
            (np.ndarray): 1D haploid profile of the data.
        """     
        # Select only cells in the specified state if isolate_state is provided
        if isolate_state is not None:
            assert self.cell_states is not None, "Cell states are not defined. Cannot isolate state."
            assert isolate_state in self.cell_states, "State {} is not defined. Cannot isolate state.".format(isolate_state)
            mask = self.cell_states == isolate_state
        # Otherwise, select all cells
        else:
            mask = np.ones(len(self.cell_labels), dtype=bool)
        # Compute the profile
        return np.nanmean(self.matrix[mask, :, :], axis=(0, 2))  # np.array of shape (ndomains,)
    
    def haploid_sort_by_row(self, isolate_state: str = None, sorter: np.ndarray = None) -> (np.ndarray, np.ndarray):
        # Placeholder
        # Creates a haploid version of the matrix, with copies stacked on top of each other, sorted by ascending value in the sorter array
        return None


# I want to put run_cellcycle and run_replication outside the class
# The reason is that these functions are specific to a particular type of data (raw counts), and thus it's better to have them as separate functions
# Indeed, I can't use them for process normalized data, for example

def compare_index(idx1: Index, idx2: Index, usechr: list) -> bool:
    """Compares two Index objects.

    Args:
        idx1 (Index): first Index object.
        idx2 (Index): second Index object.

    Returns:
        bool: True if the two Index objects are the same, False otherwise.
    """
    
    if idx1.genome.assembly != idx2.genome.assembly:
        return False
    
    # Compare the two Index objects on the chromosomes in usechr
    if np.any(idx1.chromstr[np.isin(idx1.chromstr, usechr)] != idx2.chromstr[np.isin(idx2.chromstr, usechr)]):
        return False
    if np.any(idx1.start[np.isin(idx1.chromstr, usechr)] != idx2.start[np.isin(idx2.chromstr, usechr)]):
        return False
    if np.any(idx1.end[np.isin(idx1.chromstr, usechr)] != idx2.end[np.isin(idx2.chromstr, usechr)]):
        return False
    
    return True

def impute_cellcycle(scm: SingleCellMatrix, config: dict) -> float:
    """ Imputes the cell cycle states of the cells in the SingleCellMatrix object.
    
    This method assumes that cells with lowest volume (bottom X%) are in G1,
    and cells with highest volume (top Y%) are in G2. X and Y have to be imputed.
    
    The imputation is done by optimizing the correlation coefficient between an external
    Replication Timing (RT) dataset and the RT computed from the SingleCellMatrix object.
    
    The correlation during optimization is calculated on a subset of chromosomes (usechr in config),
    e.g. only odd chromosomes, so as to avoid overfitting.

    Args:
        scm (SingleCellMatrix)
        config (dict): configuration dictionary.

    Returns:
        r (float): best optimization correlation coefficient between the RT and the cell cycle phase on the subset of chromosomes.
    """
    
    # Check that config is a dictionary
    assert isinstance(config, dict), "The input configuration must be a dictionary."
    
    # Check that the required keys are present in config
    required_keys = ['parallel', 'rt_bedfile', 'assembly', 'usechr', 'smooth', 'G1_n0', 'G1_n1', 'G2_n0', 'G2_n1']
    for key in required_keys:
        assert key in config.keys(), "The input configuration must have the key '{}'.".format(key)
    
    # create a temporary directory to store nodes' results
    temp_dir = tempfile.mkdtemp(dir=os.getcwd())
    sys.stdout.write("Temporary directory for nodes' results: {}\n".format(temp_dir))
    
    # create a Controller
    controller = Controller(config)
    
    # Read the RT data and assert that Index matches
    rt_bedfile = config['rt_bedfile']
    assembly = config['assembly']
    idx_rt = Index(rt_bedfile, genome=Genome(assembly))
    if not compare_index(scm.index, idx_rt, config['usechr']):
        raise ValueError("The Index objects of the SingleCellMatrix and the RT data do not match.")
    
    # compute all the possible G1/G2 segmentations
    segmentation = []
    # (assuming that G1 (and G2, separately) can have at most half of the cells)
    for ncell_g1 in range(config['G1_n0'], config['G1_n1']):
        for ncell_g2 in range(config['G2_n0'], config['G2_n1']):
            segmentation.append([ncell_g1, ncell_g2])
    segmentation = np.array(segmentation)
    nsegment = segmentation.shape[0]
    
    # Save segmentation, chromstr, nraw and volume to a temporary HDF5 file
    with h5py.File(os.path.join(temp_dir, 'data_for_nodes.hdf5'), 'w') as hdf5:
        hdf5.create_dataset('segmentation', data=segmentation)
        hdf5.create_dataset('chromstr', data=scm.index.chromstr.astype('S10'), dtype=np.dtype('S10'))
        hdf5.create_dataset('ncount', data=scm.matrix)
        hdf5.create_dataset('volume', data=scm.volumes)

    # set the parallel and reduce tasks
    parallel_task = partial(cellcycle.parallel_function,
                            cfg=config,
                            temp_dir=temp_dir)
    reduce_task = cellcycle.reduce_function

    # run the parallel and reduce tasks
    r, cycle = controller.map_reduce(parallel_task,
                                     reduce_task,
                                     args=np.arange(nsegment))
    
    # Delete the temporary directory and its contents
    os.system('rm -r {}'.format(temp_dir))
    
    # Update the attributes of the SingleCellMatrix object
    scm.cell_states = cycle
    
    return r

def simulate_rt(scm: SingleCellMatrix) -> np.ndarray:
    """ Simulates the Replication Timing (RT) from the SingleCellMatrix object.
    
    The RT is computed as the S phase profile divided by the detection bias.
    The detection bias is computed as the average of the G1 and G2 profiles.

    Args:
        scm (SingleCellMatrix)

    Returns:
        rt (np.ndarray): 1D haploid RT profile.
    """

    # Assert that the cell states are defined and correspond to G1, S and G2
    if scm.cell_states is None:
        raise ValueError("Cell cycle states are not defined. Cannot simulate RT.")
    if not np.all(np.isin(scm.cell_states, ['G1', 'S', 'G2'])):
        raise ValueError("Cell cycle states must be 'G1', 'S' or 'G2'.")
    
    # Calculate the bias in G1 and G2
    bias_g1 = scm.get_profile(isolate_state='G1')
    bias_g2 = scm.get_profile(isolate_state='G2')
    
    # Combine the biases and normalize them to have mean 1
    bias_g1 = bias_g1 / np.nanmean(bias_g1)
    bias_g2 = bias_g2 / np.nanmean(bias_g2)
    bias = (bias_g1 + bias_g2) / 2
    bias = bias / np.nanmean(bias)
    
    # Get the simulated RT as the S phase profile divided by the bias
    rt = scm.get_profile(isolate_state='S') / bias
    
    return rt
    