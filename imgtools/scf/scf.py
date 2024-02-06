import os
import sys
import numpy as np
import h5py
import tempfile
from functools import partial
from alabtools.utils import Genome, Index
from alabtools.parallel import Controller
from . import cellcycle


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
        
        # If the file doesn't exists, make sure that mode is write (w, w-, x)
        if not os.path.exists(h5_name) and mode not in ['w', 'w-', 'x']:
            raise FileNotFoundError("The HDF5 file does not exist. Use mode 'w', 'w-', or 'x'.")
        
        # Open the HDF5 file
        self.h5_name = h5_name
        self.h5 = h5py.File(h5_name, mode)
    
    
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
        if name in self.h5:
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
    
    
    # DEFINE PROPERTIES
    index = property(get_index, set_index, doc="Index object.")
    attrs = property(get_attrs, set_attrs, doc="Attributes.")
    cell_labels = property(get_cell_labels, set_cell_labels, doc="Cell labels.")
    cell_states = property(get_cell_states, set_cell_states, doc="Cell states.")
    volumes = property(get_volumes, set_volumes, doc="Cell volumes.")
    feature_list = property(get_feature_list, doc="List of feature matrices.")
    
    
    # DATA ADDITION FUNCTIONS
    
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
        
        self.set_cell_labels(cell_states)
    
    
    # COMPUTATION FUNCTIONS
    
    def haploid_profile(self, feature_name: str, isolate_state: str = None) -> (np.ndarray, np.ndarray):
        """ Computes a 1D haploid profile of the data, providing the mean and the standard deviation.
        If isolate_state is provided, it is computed only for the cells in that state (e.g. S phase)

        Args:
            isolate_state (str, optional): cell state to isolate. Defaults to None.

        Returns:
            mean (np.ndarray): 1D haploid profile of the data.
            std (np.ndarray): 1D haploid standard deviation of the data.
        """     
        # Select only cells in the specified state if isolate_state is provided
        if isolate_state is not None:
            if not 'cell_states' in self.h5:
                raise ValueError("Cell states are not defined. Cannot isolate state.")
            if not isolate_state in self.cell_states:
                raise ValueError("State {} is not defined. Cannot isolate state.".format(isolate_state))
            mask = self.cell_states == isolate_state
        # Otherwise, select all cells
        else:
            mask = np.ones(len(self.cell_labels), dtype=bool)
            
        # Take the feature matrix and compute the mean and standard deviation
        mat = self.get_matrix(feature_name)
        mean = np.nanmean(mat[mask, :, :], axis=(0, 2))  # np.array of shape (ndomains,)
        std = np.nanstd(mat[mask, :, :], axis=(0, 2))
        
        return mean, std
    
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

def impute_cellcycle(scm: SingleCellFeature, config: dict) -> float:
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

def simulate_rt(scm: SingleCellFeature) -> np.ndarray:
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
    